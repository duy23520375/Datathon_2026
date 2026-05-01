"""
====================================================================
Features:
- Time: year, month, day, dow, dom
- Special: tet, yr, is_p
- Seasonal: tf, df, mf (Decay weighted)
- Distance: d1, d15, payday_dow
- Cyclical: m_s, m_c, dw_s, dw_c, dm_s, dm_c
- Lags: 364, 371, 728 (Normalized)

Ensemble: 7 models (L2, Tweedie, Huber, Log, Sqrt, MLP1, MLP2)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import random
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from lunardate import LunarDate
warnings.filterwarnings('ignore')

# --- 1. SETTINGS ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
GROWTH_FACTOR = 1.12

def get_tet_date(year):
    for d in pd.date_range(f"{year}-01-01", f"{year}-03-01"):
        lunar = LunarDate.fromSolarDate(d.year, d.month, d.day)
        if lunar.month == 1 and lunar.day == 1:
            return d
    raise ValueError(f"Tet not found for year {year}")

TET_DATES = {y: get_tet_date(y) for y in range(2012, 2025)}

train_raw = pd.read_csv('sales.csv', parse_dates=['Date'])
train_raw['year'] = train_raw['Date'].dt.year
test_raw = pd.read_csv('sample_submission.csv', parse_dates=['Date'])
promos = pd.read_csv('promotions.csv', parse_dates=['start_date', 'end_date'])

# --- 2. ANCHOR ---
def get_anchor(tr, ts):
    """
    Calculates the target mean Revenue and COGS by projecting 2022 baseline
    daily values with 1.12 growth and all seasonal factors.

    :param tr: Historical training data
    :param ts: Future test dates
    :return: (Mean Revenue, Mean COGS)
    """
    df = tr.copy()
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dow'] = df['Date'].dt.dayofweek
    df['tet_offset'] = np.nan
    for y in df['year'].unique():
        if y in TET_DATES: df.loc[df['year']==y, 'tet_offset'] = (df['Date'] - TET_DATES[y]).dt.days

    annual = df.groupby('year')[['Revenue', 'COGS']].sum()
    g_r_base = (annual['Revenue'].pct_change().dropna() + 1).prod() ** (1/10)
    g_c_base = (annual['COGS'].pct_change().dropna() + 1).prod() ** (1/10)

    b_r_22 = annual.loc[2022, 'Revenue'] / 365
    b_c_22 = annual.loc[2022, 'COGS'] / 365

    decay = 0.9
    df['w'] = decay ** (df['year'].max() - df['year'])
    df['r_n'] = df['Revenue'] / df.groupby('year')['Revenue'].transform('mean')
    df['c_n'] = df['COGS'] / df.groupby('year')['COGS'].transform('mean')
    dow_p = df.groupby('dow').apply(lambda g: pd.Series(
        {
            'rf': np.average(g['r_n'], weights=g['w']),
            'cf': np.average(g['c_n'], weights=g['w'])
        }
    ))

    for c in ['rf', 'cf']:
        dow_p[c] /= dow_p[c].mean()

    df = df.merge(dow_p, on='dow')
    tet_p = df[
        (df['tet_offset'] >= -20) & (df['tet_offset'] <= 20)
    ].groupby('tet_offset').apply(lambda g: pd.Series(
        {
            'trf': np.average(g['r_n']/g['rf'], weights=g['w']),
            'tcf': np.average(g['c_n']/g['cf'], weights=g['w'])
        }
    ))

    md_p = df.merge(
        tet_p,
        on='tet_offset',
        how='left'
    ).fillna(1.0).groupby(['month', 'day']).apply(lambda g: pd.Series(
        {
            'mrf': np.average(g['r_n']/(g['rf']*g['trf']), weights=g['w']),
            'mcf': np.average(g['c_n']/(g['cf']*g['tcf']), weights=g['w'])
        }
    ))
    ts_v = ts.copy()
    ts_v['year'] = ts_v['Date'].dt.year
    ts_v['month'] = ts_v['Date'].dt.month
    ts_v['day'] = ts_v['Date'].dt.day
    ts_v['dow'] = ts_v['Date'].dt.dayofweek

    ts_v['tet_offset'] = np.nan
    for y in ts_v['year'].unique():
        ts_v.loc[ts_v['year']==y, 'tet_offset'] = (ts_v['Date'] - TET_DATES[y]).dt.days
    ts_v = (
        ts_v
        .merge(dow_p, on='dow')
        .merge(tet_p, on='tet_offset', how='left')
        .merge(md_p, on=['month', 'day'], how='left')
        .fillna(1.0)
    )
    g_r = g_r_base * GROWTH_FACTOR
    g_c = g_c_base * GROWTH_FACTOR
    ts_v['R_pred'] = b_r_22 * (g_r ** (ts_v['year']-2022)) * ts_v['rf'] * ts_v['trf'] * ts_v['mrf']
    ts_v['C_pred'] = b_c_22 * (g_c ** (ts_v['year']-2022)) * ts_v['cf'] * ts_v['tcf'] * ts_v['mcf']

    return ts_v['R_pred'].mean(), ts_v['C_pred'].mean()

TARGET_MEAN_R, TARGET_MEAN_C = get_anchor(train_raw, test_raw)

# --- 3. FEATURE ENGINE ---
def build_features(df_tr, df_ts):
    """
    Constructs the feature set with deep documentation for every column.

    --- DANH MỤC CÁC BIẾN (FEATURES) ---
    - tet: Số ngày cách Tết Nguyên Đán (Dương lịch). Giúp bắt đỉnh mua sắm trước Tết và đáy sau Tết.
    - yr: Chỉ số năm (Year Index = Năm - 2012). Giúp mô hình nhận diện xu hướng tăng trưởng dài hạn.
    - is_p: Cờ khuyến mãi (Promotion). =1 nếu ngày đó nằm trong chiến dịch khuyến mãi, =0 nếu không.
    - tf (Tet Factor): Chỉ số mùa vụ Tết. Được tính bằng trung bình có trọng số của doanh thu quanh các kỳ Tết lịch sử.
    - df (Day Factor): Chỉ số mùa vụ theo thứ trong tuần. Giúp nhận diện thói quen mua sắm cuối tuần vs ngày thường.
    - mf (Month-Day Factor): Chỉ số mùa vụ theo ngày cụ thể trong năm (ví dụ: các ngày lễ cố định như 30/4, 2/9).
    - d1: Khoảng cách tới ngày đầu/cuối tháng (Payday 1). Đại diện cho chu kỳ lĩnh lương đầu tháng.
    - d15: Khoảng cách tới ngày giữa tháng (Payday 15). Đại diện cho chu kỳ lĩnh lương giữa tháng.
    - payday_dow: Tương tác giữa ngày lĩnh lương và thứ trong tuần. Thể hiện việc mua sắm mạnh hơn nếu lương cuối tuần.
    - m_s / m_c: Mã hóa Sin/Cos của Tháng. Giúp mô hình hiểu tính chu kỳ 12 tháng (Tháng 12 gần Tháng 1).
    - dw_s / dw_c: Mã hóa Sin/Cos của Thứ. Giúp mô hình hiểu tính chu kỳ 7 ngày (Chủ nhật gần Thứ hai).
    - dm_s / dm_c: Mã hóa Sin/Cos của Ngày trong tháng. Giúp mô hình hiểu tính chu kỳ 31 ngày.
    - Revenue_lag_364/371/728: Doanh thu chuẩn hóa của 1 năm/2 năm trước. Cung cấp "trí nhớ" về hiệu suất cùng kỳ.
    """
    all_d = pd.concat([df_tr[['Date', 'Revenue', 'COGS']], df_ts[['Date']]])
    all_d['year'] = all_d['Date'].dt.year; all_d['month'] = all_d['Date'].dt.month
    all_d['day'] = all_d['Date'].dt.day; all_d['dow'] = all_d['Date'].dt.dayofweek
    all_d['tet'] = (all_d['Date'] - all_d['year'].map(TET_DATES)).dt.days

    # SF Base: Tính toán các chỉ số mùa vụ thuần túy (Seasonal Factors)
    df_tr_sf = all_d[all_d['Revenue'].notnull()].copy()
    df_tr_sf['norm'] = df_tr_sf['Revenue'] / df_tr_sf.groupby('year')['Revenue'].transform('mean')

    # tf (Tet Factor): Tính trung bình độ lệch doanh thu trong khoảng [-25, +20] ngày quanh Tết
    tp = (
        df_tr_sf[(df_tr_sf['tet'] >= -25) & (df_tr_sf['tet'] <= 20)]
        .groupby('tet')['norm']
        .mean()
        .reset_index()
        .rename(columns={'norm': 'tf'})
    )
    df_tr_sf = df_tr_sf.merge(tp, on='tet', how='left').fillna(1.0)

    # df (Day Factor): Tính độ lệch theo thứ sau khi đã loại bỏ ảnh hưởng của Tết
    dow = df_tr_sf.groupby('dow').apply(lambda x: (x['norm'] / x['tf']).mean()).reset_index().rename(columns={0: 'df'})

    # mf (Month-Day Factor): Tính độ lệch theo ngày trong năm sau khi loại bỏ cả Tết và Thứ
    md = (
        df_tr_sf
        .merge(dow, on='dow')
        .groupby(['month', 'day'])
        .apply(lambda x: (x['norm'] / (x['tf'] * x['df'])).mean())
        .reset_index()
        .rename(columns={0: 'mf'})
    )

    # Ghép các chỉ số mùa vụ vào tập dữ liệu tổng
    all_d = (
        all_d.merge(tp, on='tet', how='left')
        .merge(dow, on='dow', how='left')
        .merge(md, on=['month', 'day'], how='left')
        .fillna(1.0)
    )

    # LAGS: Doanh thu và COGS của quá khứ (đã chuẩn hóa theo trung bình năm đó)
    for target in ['Revenue', 'COGS']:
        m_map = df_tr.groupby('year')[target].mean().to_dict()
        df_target = df_tr[['Date', target]].copy()
        df_target['norm'] = df_target[target] / df_target['Date'].dt.year.map(m_map)

        for lag in [364, 371, 728]:
            all_d[f'{target}_lag_{lag}'] = (
                all_d['Date']
                .apply(lambda d: (
                    df_target.loc[df_target['Date'] == d - pd.Timedelta(days=lag), 'norm']
                    .values[0] if not df_target.loc[df_target['Date'] == d - pd.Timedelta(days=lag)].empty else 1.0
                )
                       )
            )

    # Các biến đặc trưng thời gian bổ sung (Time Features)
    all_d['dom'] = all_d['Date'].dt.day; all_d['yr'] = all_d['year'] - 2012
    all_d['is_p'] = 0
    for _, p in promos.iterrows():
        mask = ((all_d['month'] == p['start_date'].month) &
                (all_d['day'] >= p['start_date'].day) &
                (all_d['day'] <= p['end_date'].day))
        all_d.loc[mask, 'is_p'] = 1

    # Biến khoảng cách ngày lương (Payday Features)
    all_d['d1'] = np.minimum(all_d['dom']-1, 31-all_d['dom']+1) # Khoảng cách tới ngày 1
    all_d['d15'] = np.abs(all_d['dom']-15) # Khoảng cách tới ngày 15
    all_d['payday_dow'] = (all_d['d1'] * (all_d['dow'] + 1)) / 7.0 # Tương tác Thứ và Ngày lương

    # Mã hóa vòng lặp (Cyclical Encoding) - Giúp mô hình hiểu sự liên tục của thời gian
    all_d['m_s'] = np.sin(2*np.pi*all_d['month']/12); all_d['m_c'] = np.cos(2*np.pi*all_d['month']/12)
    all_d['dw_s'] = np.sin(2*np.pi*all_d['dow']/7); all_d['dw_c'] = np.cos(2*np.pi*all_d['dow']/7)
    all_d['dm_s'] = np.sin(2*np.pi*all_d['dom']/31); all_d['dm_c'] = np.cos(2*np.pi*all_d['dom']/31)

    tr_out = all_d[all_d['Date'] <= df_tr['Date'].max()].copy()
    ts_out = all_d[all_d['Date'] > df_tr['Date'].max()].copy()
    return tr_out, ts_out

print("Building Features...")
tr, ts = build_features(train_raw, test_raw)

lag_f_r = ['Revenue_lag_364', 'Revenue_lag_371', 'Revenue_lag_728']
lag_f_c = ['COGS_lag_364', 'COGS_lag_371', 'COGS_lag_728']
base_f = ['tet', 'yr', 'is_p', 'tf', 'df', 'mf', 'd1', 'd15', 'payday_dow']
f_n_base = base_f + ['m_s', 'm_c', 'dw_s', 'dw_c', 'dm_s', 'dm_c']

# --- 4. ENSEMBLE ---
def train_target(df, target, lag_feats):
    """
    Trains the 7-model ensemble for a specific target with exact v67 parameters.
    """
    tr_m = df['year'] < 2022; val_m = df['year'] == 2022
    y = df[target] / df.groupby('year')[target].transform('mean')
    f_l = base_f + lag_feats
    f_n = f_n_base + lag_feats

    models = []
    print(f"Training Stage: {target}...")
    for obj in ['regression', 'tweedie', 'huber']:
        m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, objective=obj, random_state=SEED)
        m.fit(df.loc[tr_m, f_l], y[tr_m], eval_set=[(df.loc[val_m, f_l], y[val_m])], callbacks=[lgb.early_stopping(50)])
        models.append(('raw', m))

    m_log = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=SEED+1)
    m_log.fit(df.loc[tr_m, f_l], np.log1p(y[tr_m])); models.append(('log', m_log))

    m_sqrt = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=SEED+2)
    m_sqrt.fit(df.loc[tr_m, f_l], np.sqrt(y[tr_m])); models.append(('sqrt', m_sqrt))

    sc = StandardScaler()
    X_tr = sc.fit_transform(df.loc[tr_m, f_n])
    m_n1 = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=800, random_state=SEED,
        early_stopping=True).fit(X_tr, y[tr_m])
    m_n2 = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        max_iter=800, random_state=SEED+1,
        early_stopping=True).fit(X_tr, y[tr_m])
    models.append(('mlp', m_n1)); models.append(('mlp', m_n2))
    return models, sc, f_l, f_n

mods_r, scr, fl_r, fn_r = train_target(tr, 'Revenue', lag_f_r)
mods_c, scc, fl_c, fn_c = train_target(tr, 'COGS', lag_f_c)

# --- 5. PREDICT ---
gr = (GROWTH_FACTOR ** (ts['year']-2022)).values
bv_r = train_raw[train_raw['year']==2022]['Revenue'].mean()
bv_c = train_raw[train_raw['year']==2022]['COGS'].mean()

def get_ensemble_pred(models, sc, df_ts, f_l, f_n, base_val):
    all_preds = []
    X_nn = sc.transform(df_ts[f_n])
    for mode, m in models:
        if mode == 'mlp': p = m.predict(X_nn)
        elif mode == 'raw': p = m.predict(df_ts[f_l])
        elif mode == 'log': p = np.expm1(m.predict(df_ts[f_l]))
        elif mode == 'sqrt': p = np.square(m.predict(df_ts[f_l]))
        all_preds.append(p)
    return np.mean(all_preds, axis=0) * base_val * gr

rev = get_ensemble_pred(mods_r, scr, ts, fl_r, fn_r, bv_r)
cogs = get_ensemble_pred(mods_c, scc, ts, fl_c, fn_c, bv_c)

final = test_raw[['Date']].copy(); final['Revenue'] = rev; final['COGS'] = cogs
final['Revenue'] *= (TARGET_MEAN_R / final['Revenue'].mean())
final['COGS'] *= (TARGET_MEAN_C / final['COGS'].mean())
final['Date'] = final['Date'].dt.strftime('%Y-%m-%d')
final.to_csv('final_submission.csv', index=False)
print(f"Done! final_submission.csv generated. Mean Rev: {final['Revenue'].mean():.2f}")
