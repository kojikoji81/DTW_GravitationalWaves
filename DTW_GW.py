import argparse
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from tslearn.clustering import KShape, TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 代表的な重力波イベントのデフォルト情報
KNOWN_EVENTS = {
    "GW150914": {"gps": 1126259462, "duration": 4.0, "detector": "H1"},
    "GW151226": {"gps": 1135136350, "duration": 4.0, "detector": "L1"},
    "GW170104": {"gps": 1167559936, "duration": 4.0, "detector": "H1"},
    "GW170814": {"gps": 1186848518, "duration": 4.0, "detector": "L1"},
    "GW170817": {"gps": 1186741861, "duration": 8.0, "detector": "L1"}, # 連星中性子星合体は少し長め
}

def generate_demo_data(sample_rate, duration, t0):
    """
    インターネット接続失敗時やデモ用に、クラスタリング可能なシミュレーションデータを生成します。
    チャープ信号、サイン波、およびホワイトノイズを混合します。
    """
    print("デモ用のシミュレーションデータを生成します（チャープ信号 ＋ 異なる周波数のサイン波 ＋ ノイズ）...")
    num_samples = int(sample_rate * duration)
    times = np.linspace(0, duration, num_samples)
    
    # 基本のホワイトノイズ
    signal = np.random.randn(num_samples) * 0.5
    
    # 窓幅が0.5秒であると仮定して、特定の時間帯に異なる波形を埋め込む
    # 1. 1.0〜1.5秒: 低周波サイン波 (40 Hz)
    idx1 = (times >= 1.0) & (times < 1.5)
    signal[idx1] += np.sin(2 * np.pi * 40 * times[idx1]) * 1.5
    
    # 2. 2.0〜2.5秒: チャープ信号 (30 Hz から 250 Hz に周波数が上昇するインスパイラル風波形)
    idx2 = (times >= 2.0) & (times < 2.5)
    t_chirp = times[idx2] - 2.0
    f_chirp = 30 + (250 - 30) * (t_chirp / 0.5)
    signal[idx2] += np.sin(2 * np.pi * f_chirp * t_chirp) * 2.5
    
    # 3. 3.0〜3.5秒: 高周波サイン波 (180 Hz)
    idx3 = (times >= 3.0) & (times < 3.5)
    signal[idx3] += np.sin(2 * np.pi * 180 * times[idx3]) * 1.5
    
    return TimeSeries(signal, sample_rate=sample_rate, t0=t0, unit='strain')

def parse_arguments():
    parser = argparse.ArgumentParser(description="重力波時系列データ DTW/K-Shape クラスタリングツール (高度化版)")
    parser.add_argument("--event", type=str, default=None, 
                        help=f"GWOSCの既知のイベント名。指定可能: {', '.join(KNOWN_EVENTS.keys())}")
    parser.add_argument("--gps", type=int, default=1240000000, 
                        help="開始GPS時刻 (デフォルト: 1240000000。event指定時は無視されます)")
    parser.add_argument("--duration", type=float, default=5.0, 
                        help="取得するデータ長（秒） (デフォルト: 5.0)")
    parser.add_argument("--detector", type=str, default="H1", 
                        help="検出器名: H1 (LIGO Hanford), L1 (LIGO Livingston), V1 (Virgo) (デフォルト: H1)")
    parser.add_argument("--method", type=str, choices=["kshape", "dtw", "softdtw"], default="kshape", 
                        help="クラスタリングアルゴリズム: kshape (SBD距離), dtw (DTW距離), softdtw (Soft-DTW距離)")
    parser.add_argument("--clusters", type=int, default=3, 
                        help="クラスタ数 (デフォルト: 3)")
    parser.add_argument("--win-len", type=float, default=0.5, 
                        help="分割シーケンスのウィンドウ長さ（秒） (デフォルト: 0.5)")
    parser.add_argument("--step-len", type=float, default=0.5, 
                        help="ウィンドウのスライドステップ幅（秒）。win-lenと同じならオーバーラップなし (デフォルト: 0.5)")
    parser.add_argument("--max-seq", type=int, default=30, 
                        help="クラスタリングに使用する最大シーケンス数。超えた場合はランダム抽出 (デフォルト: 30)")
    parser.add_argument("--whiten", action="store_true", default=True, 
                        help="データ前処理時に白色化 (Whitening) を適用する (デフォルト: True)")
    parser.add_argument("--no-whiten", dest="whiten", action="store_false", 
                        help="データ前処理時に白色化を適用しない")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # イベント設定のオーバーライド
    gps = args.gps
    duration = args.duration
    detector = args.detector
    
    if args.event:
        if args.event in KNOWN_EVENTS:
            event_info = KNOWN_EVENTS[args.event]
            gps = event_info["gps"]
            duration = event_info["duration"]
            detector = event_info["detector"]
            print(f"既知のイベント '{args.event}' が指定されました。設定を自動ロードします:")
            print(f"  検出器: {detector}, GPS: {gps}, データ長: {duration}秒")
        else:
            print(f"Warning: イベント '{args.event}' は未定義です。手動パラメータを使用します。")

    sample_rate = 16384  # Hz
    padding = 2.0  # バンドパスと白色化のエッジ効果を避けるための前後余白（秒）
    
    # --- 1. 重力波データの取得 ---
    print("\n[1/5] 重力波データの取得を開始します...")
    data_loaded = False
    
    try:
        # 余白を持たせてデータを取得
        print(f"GWOSCからデータを取得中: {detector}, GPS {gps - padding} から {gps + duration + padding} (前後 {padding}秒 余白)...")
        data = TimeSeries.fetch_open_data(detector, gps - padding, gps + duration + padding, sample_rate=sample_rate)
        print(f"データ取得成功: 長さ {len(data)}サンプル")
        data_loaded = True
    except Exception as e:
        print(f"GWOSCからのデータ取得中にエラーが発生しました: {e}")
        data = generate_demo_data(sample_rate, duration, gps)
    
    # --- 2. データの前処理とシーケンスへの分割 ---
    print("\n[2/5] データの前処理とシーケンス分割を行います...")
    
    # バンドパスフィルター (30-500Hz)
    data = data.bandpass(30, 500)
    
    # 白色化 (Whitening) の適用
    if args.whiten and data_loaded:
        print("白色化 (Whitening) を適用します...")
        data = data.whiten()
    elif args.whiten and not data_loaded:
        print("デモデータのため白色化はスキップします（すでにノイズレベルが調整済みのため）。")
        
    # 余白部分をカット（エッジ効果の除去）
    if data_loaded:
        data = data.crop(gps, gps + duration)
        print(f"前処理後のデータ範囲: GPS {data.t0.value:.2f} - {data.times[-1].value:.2f}, 長さ: {len(data)}サンプル")
    
    # シーケンス分割の設定
    window_samples = int(args.win_len * sample_rate)
    step_samples = int(args.step_len * sample_rate)
    
    sequences = []
    times_labels = []
    
    # ウィンドウ分割
    for i in range(0, len(data) - window_samples + 1, step_samples):
        seq = data[i : i + window_samples].value
        # 開始時間ラベル（GPS秒）
        t_start = data.t0.value + (i / sample_rate)
        
        # 標準化（振幅の大きさに依存せず、波形の「形状」を比較するため）
        if np.std(seq) > 1e-6:
            seq_norm = StandardScaler().fit_transform(seq.reshape(-1, 1)).flatten()
            sequences.append(seq_norm)
            times_labels.append(t_start)
            
    X_sequences = np.array(sequences)
    times_labels = np.array(times_labels)
    
    print(f"生成されたシーケンス数: {X_sequences.shape[0]}, シーケンス長: {X_sequences.shape[1]}")
    
    if X_sequences.shape[0] == 0:
        print("エラー: シーケンスが全く生成されませんでした。終了します。")
        return
        
    # シーケンス数の制限
    if X_sequences.shape[0] > args.max_seq:
        print(f"シーケンス数が制限値 ({args.max_seq}) を超えたため、ランダムに {args.max_seq} 個を抽出します。")
        random_indices = np.random.choice(X_sequences.shape[0], args.max_seq, replace=False)
        random_indices.sort()  # 時系列順をある程度保つためにソート
        X_sequences = X_sequences[random_indices]
        times_labels = times_labels[random_indices]
        
    # --- 3. クラスタリングの実行 ---
    print(f"\n[3/5] クラスタリング ({args.method.upper()}) を実行します...")
    n_clusters = min(args.clusters, X_sequences.shape[0])
    print(f"設定クラスタ数: {n_clusters}")
    
    if args.method == "kshape":
        model = KShape(n_clusters=n_clusters, random_state=42, verbose=True, max_iter=15)
    elif args.method == "dtw":
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42, verbose=True, max_iter=10, n_jobs=-1)
    elif args.method == "softdtw":
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="softdtw", metric_params={"gamma": 0.1}, random_state=42, verbose=True, max_iter=10, n_jobs=-1)
        
    model.fit(X_sequences)
    labels = model.labels_
    centroids = model.cluster_centers_
    print("クラスタリング処理が完了しました。")
    
    # --- 4. 結果の可視化 (Matplotlib) ---
    print("\n[4/5] Matplotlib を用いて静的グラフを保存します...")
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3 * n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]
        
    colors = plt.colormaps['tab10']
    
    for yi in range(n_clusters):
        ax = axes[yi]
        cluster_color = colors(yi)
        cluster_indices = np.where(labels == yi)[0]
        
        # クラスタに属する個々のシーケンスをプロット
        for idx in cluster_indices:
            ax.plot(X_sequences[idx].ravel(), color=cluster_color, alpha=0.35, linewidth=1)
            
        # セントロイドを強調プロット
        ax.plot(centroids[yi].ravel(), color=cluster_color, linewidth=2.5, linestyle='--',
                label=f"Cluster {yi + 1} Centroid (Count: {len(cluster_indices)})")
        
        ax.set_xlim(0, X_sequences.shape[1] - 1)
        ax.set_ylabel("Standardized Strain")
        ax.set_title(f"Cluster {yi + 1} Pattern Group")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.xlabel("Index (Samples)")
    title_suffix = f" ({args.event})" if args.event else f" (GPS: {gps})"
    plt.suptitle(f"Gravitational Wave Sequence Clustering using {args.method.upper()}{title_suffix}", y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "gw_dtw_clustering.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"静的グラフを保存しました: {png_path}")
    
    # --- 5. 結果の可視化 (Plotly - インタラクティブ) ---
    print("\n[5/5] Plotly を用いてインタラクティブグラフを生成します...")
    plotly_fig = make_subplots(rows=n_clusters, cols=1, 
                               shared_xaxes=True,
                               subplot_titles=[f"Cluster {i+1} (Interactive View)" for i in range(n_clusters)],
                               vertical_spacing=0.08)
    
    colors_hex = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    
    for yi in range(n_clusters):
        cluster_color = colors_hex[yi % len(colors_hex)]
        cluster_indices = np.where(labels == yi)[0]
        
        # 各シーケンスを追加
        for idx in cluster_indices:
            t_lbl = times_labels[idx]
            plotly_fig.add_trace(
                go.Scatter(
                    y=X_sequences[idx].ravel(),
                    mode='lines',
                    line=dict(color=cluster_color, width=1),
                    opacity=0.3,
                    name=f"Seq {idx} (GPS: {t_lbl:.3f})",
                    hoverinfo="y+name"
                ),
                row=yi+1, col=1
            )
            
        # セントロイドを追加
        plotly_fig.add_trace(
            go.Scatter(
                y=centroids[yi].ravel(),
                mode='lines',
                line=dict(color=cluster_color, width=3, dash='dash'),
                name=f"Centroid {yi+1} (Total: {len(cluster_indices)})",
                hoverinfo="y+name"
            ),
            row=yi+1, col=1
        )
        
        plotly_fig.update_yaxes(title_text="Strain (Std)", row=yi+1, col=1)
        
    plotly_fig.update_xaxes(title_text="Sample Index", row=n_clusters, col=1)
    plotly_fig.update_layout(
        height=250 * n_clusters + 150,
        title_text=f"Gravitational Wave Pattern Clustering ({args.method.upper()})<br>Event/GPS: {args.event or gps} | Detector: {detector}",
        showlegend=False,
        template="plotly_white"
    )
    
    html_path = os.path.join(output_dir, "gw_dtw_clustering_interactive.html")
    plotly_fig.write_html(html_path)
    print(f"インタラクティブグラフを保存しました: {html_path}")
    
    # ターミナルへ概要を出力
    print("\n" + "="*40)
    print("      クラスタリング実行サマリー")
    print("="*40)
    print(f"分析イベント/GPS : {args.event or gps}")
    print(f"使用検出器       : {detector}")
    print(f"適用アルゴリズム : {args.method.upper()}")
    print(f"総シーケンス数   : {X_sequences.shape[0]}")
    for i in range(n_clusters):
        c_count = np.sum(labels == i)
        print(f"  - クラスタ {i+1} : {c_count} サンプル")
    print("="*40)

if __name__ == "__main__":
    main()