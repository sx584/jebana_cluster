# freqai/prediction_models/LSTMRegressor.py
import numpy as np
import pandas as pd
import torch
from typing import Any
from pandas import DataFrame

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import DefaultPyTorchDataConvertor, PyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchModelTrainer

from freqtrade.freqai.prediction_models.LSTMModel import LSTMModel  # Pfad anpassen
from torch.serialization import add_safe_globals
add_safe_globals([PyTorchModelTrainer, LSTMModel])


class PyTorchLSTMRegressor(BasePyTorchRegressor):
    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})

        self.learning_rate: float = config.get("learning_rate", 1e-3)  # Höher für Lernen
        self.seq_len: int = config.get("seq_len", 16)  # Noch kürzer
        self.dropout: float = config.get("dropout", 0.1)  # Weniger Dropout

        default_model_kwargs = {
            "hidden_size": 128,  # Kleiner für Stabilität
            "num_layers": 1,  # Weniger Layers = stabiler
            "bidirectional": False,
            "use_last_timestep": True,
        }
        
        # Weniger aggressives Gradient Clipping
        default_trainer_kwargs = {
            "max_grad_norm": 1.0,  # Weniger aggressives Clipping
            "early_stopping_patience": 20,
        }
        user_trainer_kwargs = config.get("trainer_kwargs", {})
        self.trainer_kwargs: dict[str, Any] = {**default_trainer_kwargs, **user_trainer_kwargs}
        user_model_kwargs = config.get("model_kwargs", {})
        self.model_kwargs: dict[str, Any] = {**default_model_kwargs, **user_model_kwargs}

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        n_features_total = data_dictionary["train_features"].shape[-1]
        # FreqAI liefert Sequenzen bereits geflattet: (seq_len * features_per_step)
        # Für das LSTM benötigen wir die Feature-Anzahl pro Zeitschritt
        features_per_step = max(1, int(n_features_total // self.seq_len))
        self.features_per_step = features_per_step
        
        # Target-Normalisierung für bessere LSTM-Konvergenz
        train_targets = data_dictionary["train_labels"]
        if hasattr(train_targets, 'values'):
            train_targets = train_targets.values
        
        # Weniger aggressives Target-Clipping
        clip_threshold = 5.0  # 5-Sigma Clipping (weniger aggressiv)
        train_targets_clipped = np.clip(train_targets, 
                                      train_targets.mean() - clip_threshold * train_targets.std(),
                                      train_targets.mean() + clip_threshold * train_targets.std())
        
        # Z-Score Normalisierung der geclippten Targets
        target_mean = train_targets_clipped.mean()
        target_std = train_targets_clipped.std() + 1e-8  # Avoid division by zero
        self.target_mean = target_mean
        self.target_std = target_std
        
        # Normalisiere alle Targets
        data_dictionary["train_labels"] = (data_dictionary["train_labels"] - target_mean) / target_std
        if "test_labels" in data_dictionary:
            data_dictionary["test_labels"] = (data_dictionary["test_labels"] - target_mean) / target_std

        model = LSTMModel(
            input_dim=features_per_step,
            seq_len=self.seq_len,
            dropout=self.dropout,
            **self.model_kwargs,
        ).to(self.device)

        # Gradient Clipping für LSTM-Stabilität
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = torch.nn.MSELoss()  # Zurück zu MSE für Regression

        trainer = self.get_init_model(dk.pair)
        if trainer is None:
            trainer = PyTorchModelTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                data_convertor=self.data_convertor,
                tb_logger=self.tb_logger,
                **self.trainer_kwargs,
            )

        trainer.fit(data_dictionary, self.splits)
        return trainer
    
    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs):
        """
        Predict mit Target-Denormalisierung
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Standard FreqAI Predict-Logic
        dk.find_features(unfiltered_df)
        filtered_df, _ = dk.filter_features(
            unfiltered_df, dk.training_features_list, training_filter=False
        )
        
        # Coerce zu float32
        filtered_df = self._coerce_to_float32(filtered_df)
        
        # Predict: reshape zurück zu (B, T, C) falls nötig
        X_np = filtered_df.values.astype("float32", copy=False)
        # Falls X flach ist, teile in (seq_len, features_per_step)
        if X_np.shape[1] == self.seq_len * getattr(self, 'features_per_step', X_np.shape[1]):
            try:
                X_np = X_np.reshape(X_np.shape[0], self.seq_len, self.features_per_step)
            except Exception:
                pass
        X = torch.from_numpy(X_np).to(self.device)
        
        # Modell aus Trainer extrahieren
        if hasattr(self.model, 'model'):
            model = self.model.model  # PyTorchModelTrainer.model
        else:
            model = self.model  # Falls direkt das Modell
        
        model.eval()
        with torch.no_grad():
            predictions = model(X).detach().cpu().numpy()
        
        # Denormalisiere Predictions zurück zu Original-Skala
        if hasattr(self, 'target_mean') and hasattr(self, 'target_std'):
            predictions = predictions * self.target_std + self.target_mean
        
        # Labels korrekt mappen
        label = dk.label_list[0] if len(dk.label_list) == 1 else dk.label_list
        pred_df = DataFrame(predictions, index=filtered_df.index,
                    columns=[label] if isinstance(label, str) else label)
        # do_preds muss 1D (np.ndarray/Series) sein – kein DataFrame!
        finite_mask = np.isfinite(predictions).all(axis=1) if predictions.ndim == 2 else np.isfinite(predictions)
        do_preds = finite_mask.astype(bool)
        return pred_df, do_preds
    
    def _coerce_to_float32(self, df: DataFrame) -> DataFrame:
        """Helper für Data-Type-Konvertierung"""
        # Numerische Spalten auswählen
        num_cols = df.select_dtypes(include=["number"]).columns
        if len(num_cols) == 0:
            return df
        
        # Inf -> NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Extremwerte clippen
        f32 = np.finfo(np.float32)
        df[num_cols] = df[num_cols].clip(lower=-f32.max, upper=f32.max)
        df[num_cols] = df[num_cols].mask(df[num_cols].abs() < f32.tiny, 0.0)
        
        # Cast zu float32
        with np.errstate(all='ignore'):
            df[num_cols] = df[num_cols].astype("float32")
        
        # NaNs füllen
        df[num_cols] = df[num_cols].fillna(0.0)
        return df
