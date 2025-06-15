import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from tslearn.clustering import KShape
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import datetime # 日付時刻モジュールをインポート
import os # ファイルパス操作のためにインポート

# --- 1. 重力波データの取得（GWOSCからサンプルデータをダウンロード） ---
print("1. 重力波データの取得を開始します...")

# データ取得期間の設定を極端に短くして、生成されるシーケンス数を減らす
duration = 5 # 取得する秒数を極端に短縮 (5秒)
t0 = 1240000000
detector = 'H1'
channel = f'{detector}:GWOSC-16KHZ_R1_STRAIN'
sample_rate = 16384 # Hz

try:
    data = TimeSeries.fetch_open_data(detector, t0, t0 + duration, sample_rate=sample_rate)
    print(f"データ取得完了: 期間 {data.t0.value:.0f} - {data.tf.value:.0f} (GPS時), 長さ {len(data)}サンプル")
except Exception as e:
    print(f"データ取得中にエラーが発生しました。インターネット接続または指定期間を確認してください: {e}")
    print("デモ用にランダムデータを生成して続行します。")
    target_num_sequences = 10 # 目標とするシーケンスの数 (10個に設定)
    window_length_seconds_for_demo = 0.5
    window_length_samples_for_demo = int(window_length_seconds_for_demo * sample_rate)
    
    num_samples_needed = target_num_sequences * window_length_samples_for_demo
    
    data = TimeSeries(np.random.randn(num_samples_needed), sample_rate=sample_rate, t0=t0, unit='strain')
    print(f"デモ用ランダムデータ生成完了: 長さ {len(data)}サンプル")


# --- 2. データの前処理とシーケンスへの分割 ---
print("2. データの前処理とシーケンスへの分割を行います...")

# ノイズ除去のためのバンドパスフィルター (例: 30-500Hz)
data = data.bandpass(30, 500)

# クラスタリングのための時系列シーケンスの作成
window_length_seconds = 0.5 # 各シーケンスの長さ (0.5秒)
window_length_samples = int(window_length_seconds * data.sample_rate.value)

step_size_samples = window_length_samples # オーバーラップなし (シーケンス数が減る)

sequences = []
for i in range(0, len(data) - window_length_samples + 1, step_size_samples):
    seq = data[i : i + window_length_samples].value
    
    if np.std(seq) > 1e-6:
        seq = StandardScaler().fit_transform(seq.reshape(-1, 1)).flatten()
    else:
        seq = np.zeros_like(seq)
    sequences.append(seq)

X_sequences = np.array(sequences)

print(f"生成されたシーケンス数: {X_sequences.shape[0]}, 各シーケンスの長さ: {X_sequences.shape[1]}")

if X_sequences.shape[0] < 10:
    print(f"Warning: 目標の10シーケンスを生成できませんでした（生成数: {X_sequences.shape[0]}）。データ長またはウィンドウ設定を再確認してください。")
    if X_sequences.shape[0] == 0:
        print("シーケンスが全く生成されませんでした。終了します。")
        exit()

if X_sequences.shape[0] > 10:
    random_indices = np.random.choice(X_sequences.shape[0], 10, replace=False)
    X_sequences = X_sequences[random_indices]
    print(f"シーケンス数が10個を超えたため、ランダムに10個選択しました。最終シーケンス数: {X_sequences.shape[0]}")


# --- 3. DTWクラスタリングの実行 ---
print("3. DTWクラスタリングを実行します...")

n_clusters = min(3, X_sequences.shape[0])
if n_clusters == 0:
    print("クラスタリング可能なシーケンスがありません。")
    exit()

kmeans_dtw = KShape(n_clusters=n_clusters, random_state=0, verbose=True, max_iter=10)

try:
    kmeans_dtw.fit(X_sequences)
    labels = kmeans_dtw.labels_
    centroids = kmeans_dtw.cluster_centers_
    print("クラスタリング完了。")
except Exception as e:
    print(f"クラスタリング中にエラーが発生しました。データを確認してください: {e}")
    print("ヒント: サンプル数が非常に多い場合、max_iterを増やすと時間がかかります。また、クラスタリングできないほどデータが少ない可能性もあります。")
    exit()


# --- 4. 結果の可視化 ---
print("4. 結果を可視化します...")

plt.figure(figsize=(15, 4 * n_clusters))

# 各クラスタに割り当てる色を定義
colors = plt.cm.get_cmap('viridis', n_clusters)

# 各サブプロットで個々の波形とセントロイドをプロット
for yi in range(n_clusters):
    ax = plt.subplot(n_clusters, 1, 1 + yi)
    
    current_cluster_color = colors(yi)
    for xx in X_sequences[labels == yi]:
        ax.plot(xx.ravel(), color=current_cluster_color, alpha=.5)
    
    ax.plot(centroids[yi].ravel(), color=current_cluster_color, linewidth=2, linestyle='--',
            label=f"Cluster {yi + 1} Centroid (Count: {np.sum(labels == yi)})")
    
    ax.set_xlim(0, X_sequences.shape[1])
    ax.set_ylim(X_sequences.min() * 1.1, X_sequences.max() * 1.1) 
    ax.set_title(f"Cluster {yi + 1}")
    ax.legend(loc='upper right')

plt.tight_layout()
plt.suptitle("Gravitational Wave Time Series Clustering using K-Shape (Limited Samples)", y=1.02, fontsize=16)

# --- 画像の保存 ---
# 保存先のディレクトリとファイル名を定義
output_dir = "plots" # 'plots'というディレクトリに保存

now = datetime.datetime.now()
filename = "gw_dtw_clustering.png"
filepath = os.path.join(output_dir, filename)

plt.savefig(filepath, dpi=300, bbox_inches='tight') # DPIを高く設定し、余白を調整して保存
print(f"\nグラフを {filepath} に保存しました。")

plt.show() # グラフを表示

print("\n--- クラスタリング結果の概要 ---")
for i in range(n_clusters):
    print(f"クラスタ {i+1}: {np.sum(labels == i)} サンプル")