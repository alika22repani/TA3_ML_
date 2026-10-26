import numpy as np
import pandas as pd

FEATURES = ['air_temp', 'process_temp', 'rpm', 'torque', 'tool_wear',
            'temp_diff', 'power_w', 'type_M', 'type_H']

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df wajib punya kolom: type, air_temp, process_temp, rpm, torque, tool_wear."""
    X = pd.DataFrame(index=df.index)
    for c in ['air_temp', 'process_temp', 'rpm', 'torque', 'tool_wear']:
        X[c] = df[c].astype(float)
    # Fitur turunan berbasis fisika (membantu model linear menangkap pola non-linear)
    X['temp_diff'] = X['process_temp'] - X['air_temp']          # penyebab Heat Dissipation Failure
    X['power_w'] = X['torque'] * X['rpm'] * 2 * np.pi / 60      # penyebab Power Failure
    X['type_M'] = (df['type'] == 'M').astype(int)
    X['type_H'] = (df['type'] == 'H').astype(int)
    return X[FEATURES]
