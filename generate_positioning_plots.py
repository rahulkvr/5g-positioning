import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def parse_matlab_matrix(text: str) -> np.ndarray:
    """Parse strings like '[1 2 3;4 5 6]' into a float matrix."""
    if isinstance(text, str):
        stripped = text.strip()
        if not stripped or stripped.lower() in {'nan', '[]'}:
            return np.empty((0, 0))
        stripped = stripped.strip('[]')
        rows = stripped.split(';')
        data = []
        for row in rows:
            row = row.strip()
            if not row:
                continue
            parts = row.replace(',', ' ').split()
            try:
                data.append([float(part) for part in parts])
            except ValueError:
                return np.empty((0, 0))
        return np.array(data, dtype=float)
    if isinstance(text, (list, tuple, np.ndarray)):
        return np.asarray(text, dtype=float)
    return np.empty((0, 0))


def parse_matlab_vector(text: str) -> np.ndarray:
    """Parse strings like '[1 4 7]' into an integer array."""
    if isinstance(text, str):
        stripped = text.strip()
        if not stripped or stripped.lower() in {'nan', '[]'}:
            return np.array([], dtype=int)
        stripped = stripped.strip('[]')
        parts = stripped.replace(';', ' ').split()
        values = []
        for part in parts:
            try:
                values.append(int(round(float(part))))
            except ValueError:
                continue
        return np.array(values, dtype=int)
    if isinstance(text, (list, tuple, np.ndarray)):
        return np.asarray(text, dtype=int)
    return np.array([], dtype=int)


def compute_row_gdop(tx_positions: np.ndarray, anchor_indices: np.ndarray, ue_xyz: np.ndarray) -> float:
    """Compute GDOP for a UE location and chosen anchor indices."""
    if tx_positions.size == 0 or anchor_indices.size < 4:
        return np.nan
    anchor_indices = anchor_indices - 1  # convert MATLAB 1-based to 0-based
    valid = (anchor_indices >= 0) & (anchor_indices < tx_positions.shape[0])
    anchor_indices = anchor_indices[valid]
    if anchor_indices.size < 4:
        return np.nan
    anchors = tx_positions[anchor_indices]
    vectors = anchors - ue_xyz
    distances = np.linalg.norm(vectors, axis=1)
    if np.any(distances < 1e-3):
        return np.nan
    directions = vectors / distances[:, None]
    ref_idx = int(np.argmin(distances))
    others = np.delete(directions, ref_idx, axis=0)
    H = others - directions[ref_idx]
    if H.shape[0] < 3:
        return np.nan
    try:
        Q = np.linalg.inv(H.T @ H)
    except np.linalg.LinAlgError:
        return np.nan
    return float(np.sqrt(np.trace(Q)))


def ensure_directory(path: Path) -> Path:
    path.mkdir(exist_ok=True)
    return path


csv_path = Path('positioning_sweep_results.csv')
if not csv_path.exists():
    raise SystemExit('positioning_sweep_results.csv not found')

df = pd.read_csv(csv_path)
# Basic cleaning/convenience columns
df['SNR'] = df['AppliedSNR_dB']
df['UKF'] = df['UKFEnabled'].astype(bool)
df['Scenario'] = df['ScenarioTag']
df['ScenarioLabel'] = df.apply(lambda row: f"SNR {row['SNR']:.0f} dB | {'UKF On' if row['UKF'] else 'UKF Off'}", axis=1)

def gdop_from_row(row):
    try:
        tx_positions = parse_matlab_matrix(row['TxPositions'])
        anchors = parse_matlab_vector(row['SelectedAnchors'])
        ue = np.array([row['TrueX'], row['TrueY'], row['TrueZ']], dtype=float)
        return compute_row_gdop(tx_positions, anchors, ue)
    except Exception:
        return np.nan


df['GDOP_Selected'] = df.apply(gdop_from_row, axis=1)
gdop_valid = df[np.isfinite(df['GDOP_Selected']) & df['HorizontalError'].notna()].copy()
gdop_pearson = np.nan
gdop_spearman = np.nan

plot_dir = ensure_directory(Path('plots'))

ukf_labels = {False: 'UKF Off', True: 'UKF On'}
colors = {False: '#1f77b4', True: '#d62728'}
tab20 = plt.cm.get_cmap('tab20', 20)
scenario_palette = {
    scenario: tab20(i % 20)
    for i, scenario in enumerate(sorted(df['Scenario'].unique()))
}

# 1. Position Error CDFs (step + smoothed)
sorted_errors = {}
fig_step, ax_step = plt.subplots(figsize=(7, 5))
for scenario, group in df.groupby('Scenario'):
    data = np.sort(group['HorizontalError'].dropna().values)
    if data.size == 0:
        continue
    sorted_errors[scenario] = data
    y = (np.arange(data.size) + 1) / data.size
    line, = ax_step.step(data, y, where='post', color=scenario_palette.get(scenario, '#333333'))
    line.set_label(scenario)
ax_step.set_xlabel('Horizontal Error (m)')
ax_step.set_ylabel('Empirical CDF')
ax_step.set_title('Horizontal Error CDF by Scenario')
ax_step.grid(True, linestyle='--', alpha=0.5)
ax_step.legend(title='Scenario', fontsize=8)
fig_step.tight_layout()
fig_step.savefig(plot_dir / 'position_error_cdf.png', dpi=200)
plt.close(fig_step)

fig_smooth, ax_smooth = plt.subplots(figsize=(7, 5))
for scenario, data in sorted_errors.items():
    if data.size < 2:
        continue
    y = (np.arange(data.size) + 1) / data.size
    x_smooth = np.linspace(data.min(), data.max(), 400)
    y_smooth = np.interp(x_smooth, data, y)
    ax_smooth.plot(x_smooth, y_smooth, label=scenario, color=scenario_palette.get(scenario, '#333333'))
ax_smooth.set_xlabel('Horizontal Error (m)')
ax_smooth.set_ylabel('Empirical CDF')
ax_smooth.set_title('Horizontal Error CDF by Scenario (Smoothed)')
ax_smooth.grid(True, linestyle='--', alpha=0.5)
ax_smooth.legend(title='Scenario', fontsize=8)
fig_smooth.tight_layout()
fig_smooth.savefig(plot_dir / 'position_error_cdf_smooth.png', dpi=200)
plt.close(fig_smooth)

# 2. Error vs SNR Summary (mean and 95th percentile)
summary = df.groupby(['SNR', 'UKF']).agg(
    mean_error=('HorizontalError', 'mean'),
    p95_error=('HorizontalError', lambda x: np.percentile(x, 95))
).reset_index()
summary.sort_values(['SNR', 'UKF'], inplace=True)

fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
for ukf_state, group in summary.groupby('UKF'):
    axes[0].plot(group['SNR'], group['mean_error'], marker='o', color=colors[ukf_state], label=ukf_labels[ukf_state])
    axes[1].plot(group['SNR'], group['p95_error'], marker='o', color=colors[ukf_state], label=ukf_labels[ukf_state])
axes[0].set_ylabel('Mean Horizontal Error (m)')
axes[0].set_title('Error vs SNR Summary')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[1].set_ylabel('95th Percentile Error (m)')
axes[1].set_xlabel('SNR (dB)')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()
fig.tight_layout()
fig.savefig(plot_dir / 'error_vs_snr.png', dpi=200)
plt.close(fig)

# 3. Violin by UE Position (horizontal error)
positions = sorted(df['PositionIndex'].unique())
fig, ax = plt.subplots(figsize=(9, 5))
width = 0.35
for i, pos in enumerate(positions):
    base_x = i + 1
    for j, ukf_state in enumerate([False, True]):
        data = df[(df['PositionIndex'] == pos) & (df['UKF'] == ukf_state)]['HorizontalError'].values
        if data.size == 0:
            continue
        xpos = base_x + (-width / 2 if j == 0 else width / 2)
        vp = ax.violinplot([data], positions=[xpos], widths=width * 0.9, showmeans=True, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(colors[ukf_state])
            body.set_alpha(0.6)
        vp['cmeans'].set_color('k')
        ax.scatter(np.full(data.size, xpos), data, color=colors[ukf_state], alpha=0.4, s=8)
ax.set_xticks(range(1, len(positions) + 1))
ax.set_xticklabels([f'Pos {p}' for p in positions])
ax.set_xlim(0.5, len(positions) + 0.5)
ax.set_ylabel('Horizontal Error (m)')
ax.set_title('Horizontal Error Distribution by UE Position')
legend_handles = [plt.Line2D([0], [0], color=colors[k], lw=6, alpha=0.6, label=ukf_labels[k]) for k in [False, True]]
ax.legend(handles=legend_handles)
ax.grid(True, linestyle='--', alpha=0.4)
fig.tight_layout()
fig.savefig(plot_dir / 'error_by_position_violin.png', dpi=200)
plt.close(fig)

# 4. CDF shift per trial count (Scenario with most samples)
scenario_counts = df['Scenario'].value_counts()
base_scenario = scenario_counts.idxmax()
base_df = df[df['Scenario'] == base_scenario].sort_values('HorizontalError')
fig, ax = plt.subplots(figsize=(7, 5))
for label, subset in [('First 10', base_df.head(10)), ('All Samples', base_df)]:
    data = np.sort(subset['HorizontalError'].values)
    if data.size == 0:
        continue
    y = (np.arange(data.size) + 1) / data.size
    ax.step(data, y, where='post', label=f'{base_scenario} - {label}')
ax.set_xlabel('Horizontal Error (m)')
ax.set_ylabel('Empirical CDF')
ax.set_title('CDF Shift vs. Sample Count')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(plot_dir / 'cdf_shift_trials.png', dpi=200)
plt.close(fig)

# 4b. GDOP vs horizontal error
if not gdop_valid.empty:
    fig, ax = plt.subplots(figsize=(7, 5))
    scenario_count = gdop_valid['Scenario'].nunique()
    for scenario, group in gdop_valid.groupby('Scenario'):
        color = scenario_palette.get(scenario, (0.35, 0.35, 0.35, 0.8))
        ax.scatter(
            group['GDOP_Selected'],
            group['HorizontalError'],
            alpha=0.55,
            s=24,
            label=scenario if scenario_count <= 8 else None,
            color=color
        )
    x_vals = gdop_valid['GDOP_Selected'].values
    y_vals = gdop_valid['HorizontalError'].values
    if x_vals.size >= 2:
        try:
            coeffs = np.polyfit(x_vals, y_vals, 1)
            x_fit = np.linspace(x_vals.min(), x_vals.max(), 200)
            y_fit = np.polyval(coeffs, x_fit)
            ax.plot(x_fit, y_fit, color='black', linewidth=2, label='Linear fit')
        except np.linalg.LinAlgError:
            pass
        gdop_pearson = float(pd.Series(x_vals).corr(pd.Series(y_vals), method='pearson'))
        gdop_spearman = float(pd.Series(x_vals).corr(pd.Series(y_vals), method='spearman'))
        ax.text(
            0.02,
            0.98,
            f'Pearson r = {gdop_pearson:.2f}\nSpearman ρ = {gdop_spearman:.2f}',
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray')
        )
    ax.set_xlabel('GDOP (Selected Anchors)')
    ax.set_ylabel('Horizontal Error (m)')
    ax.set_title('GDOP vs Horizontal Error')
    ax.grid(True, linestyle='--', alpha=0.4)
    if scenario_count <= 8:
        ax.legend(fontsize=8, title='Scenario')
    fig.tight_layout()
    fig.savefig(plot_dir / 'gdop_vs_error_scatter.png', dpi=200)
    plt.close(fig)

    # Binned summary of GDOP vs error
    bin_count = min(12, max(4, int(np.sqrt(gdop_valid.shape[0] / 5))))
    bin_edges = np.linspace(gdop_valid['GDOP_Selected'].min(), gdop_valid['GDOP_Selected'].max(), bin_count + 1)
    if np.all(np.isfinite(bin_edges)) and bin_edges.size >= 5:
        gdop_binned = gdop_valid.copy()
        gdop_binned['GDOPBin'] = pd.cut(gdop_binned['GDOP_Selected'], bins=bin_edges, include_lowest=True, duplicates='drop')
        bin_stats = gdop_binned.dropna(subset=['GDOPBin']).groupby('GDOPBin')['HorizontalError'].agg(['mean', 'median', 'count']).reset_index()
        if not bin_stats.empty:
            centers = np.array([(interval.left + interval.right) / 2 for interval in bin_stats['GDOPBin']])
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(centers, bin_stats['mean'], marker='o', label='Mean error', color='#1f77b4')
            ax.plot(centers, bin_stats['median'], marker='s', label='Median error', color='#ff7f0e')
            ax.set_xlabel('GDOP (bin center)')
            ax.set_ylabel('Horizontal Error (m)')
            ax.set_title('Error vs GDOP (Binned Summary)')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()
            ax2 = ax.twinx()
            width = 0.6 * np.diff(centers).min() if centers.size > 1 else 0.5
            ax2.bar(centers, bin_stats['count'], width=width, alpha=0.25, color='gray')
            ax2.set_ylabel('Samples per bin')
            fig.tight_layout()
            fig.savefig(plot_dir / 'gdop_vs_error_binned.png', dpi=200)
            plt.close(fig)

# 5. Heatmap: mean horizontal error by scenario & position index
heatmap_data = df.groupby(['Scenario', 'PositionIndex'])['HorizontalError'].mean().unstack().reindex(sorted(df['Scenario'].unique()))
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
ax.set_xticks(range(len(heatmap_data.columns)))
ax.set_xticklabels([f'P{int(p)}' for p in heatmap_data.columns])
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)
ax.set_xlabel('UE Position Index')
ax.set_ylabel('Scenario')
ax.set_title('Mean Horizontal Error Heatmap')
fig.colorbar(im, ax=ax, label='Error (m)')
fig.tight_layout()
fig.savefig(plot_dir / 'error_heatmap_scenario_position.png', dpi=200)
plt.close(fig)

# 6. Scatter: correlation metric vs error
fig, ax = plt.subplots(figsize=(7, 5))
for snr, group in df.groupby('SNR'):
    ax.scatter(group['MaxCorrAverage'], group['HorizontalError'], alpha=0.6, label=f'{snr:.0f} dB')
ax.set_xlabel('MaxCorrAverage')
ax.set_ylabel('Horizontal Error (m)')
ax.set_title('Correlation Metric vs Horizontal Error')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(title='SNR')
fig.tight_layout()
fig.savefig(plot_dir / 'corr_vs_error.png', dpi=200)
plt.close(fig)

# 7. Histogram of TDOA correlation metric per SNR
fig, ax = plt.subplots(figsize=(7, 5))
for snr, group in df.groupby('SNR'):
    data = group['MaxCorrAverage'].dropna().values
    if data.size == 0:
        continue
    ax.hist(data, bins=np.linspace(0, 1, 21), histtype='step', linewidth=2, label=f'SNR {snr:.0f} dB')
ax.set_xlabel('MaxCorrAverage')
ax.set_ylabel('Count')
ax.set_title('TDOA Correlation Metric Distribution by SNR')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
fig.tight_layout()
fig.savefig(plot_dir / 'corr_hist.png', dpi=200)
plt.close(fig)

# 8. Solver fallback frequency by scenario
fallback_summary = df.groupby('Scenario')['CentroidFallback'].sum()
total_counts = df['Scenario'].value_counts()
rate = (fallback_summary / total_counts).fillna(0)
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(rate.index, rate.values, color='#9467bd')
ax.set_ylabel('Fallback Rate (fraction of trials)')
ax.set_title('Centroid Fallback Frequency by Scenario')
ax.set_ylim(0, 1)
for tick in ax.get_xticklabels():
    tick.set_rotation(30)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig(plot_dir / 'solver_fallback_rate.png', dpi=200)
plt.close(fig)

# 9. Path loss vs error scatter (proxy for propagation quality)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df['PathLossMean'], df['HorizontalError'], c=df['SNR'], cmap='coolwarm', alpha=0.6)
ax.set_xlabel('Mean Path Loss (dB)')
ax.set_ylabel('Horizontal Error (m)')
ax.set_title('Path Loss vs Horizontal Error')
cb = fig.colorbar(ax.collections[0], ax=ax, label='SNR (dB)')
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()
fig.savefig(plot_dir / 'pathloss_vs_error.png', dpi=200)
plt.close(fig)

# 10. Estimated vs true positions inside STL scene
mesh_path = Path('train_station.stl')
scale_factor = 12.0  # From PositioningConfig
if mesh_path.exists() and trimesh is not None:
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        mesh.apply_scale(scale_factor)
        mesh_verts = mesh.vertices
        mesh_faces = mesh.faces
    except Exception as exc:
        mesh = None
        print(f'Warning: Failed to load STL mesh ({exc}). Skipping environment overlay.')
else:
    mesh = None
    if not mesh_path.exists():
        print('train_station.stl not found. Skipping 3D environment visualization.')
    elif trimesh is None:
        print('trimesh package not available. Install via pip to enable STL visualization.')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
if mesh is not None:
    mesh_collection = Poly3DCollection(mesh_verts[mesh_faces], alpha=0.05, facecolor='#888888', edgecolor='none')
    ax.add_collection3d(mesh_collection)
    bounds = np.array([mesh_verts.min(axis=0), mesh_verts.max(axis=0)])
else:
    bounds = np.vstack([
        df[['TrueX', 'TrueY', 'TrueZ']].min().values,
        df[['TrueX', 'TrueY', 'TrueZ']].max().values
    ])

true_pts = df[['TrueX', 'TrueY', 'TrueZ', 'PositionIndex']].drop_duplicates().sort_values('PositionIndex')
true_xyz = true_pts[['TrueX', 'TrueY', 'TrueZ']].values
ax.scatter(true_xyz[:, 0], true_xyz[:, 1], true_xyz[:, 2],
           c='k', s=70, label='True UE path', depthshade=False)
ax.plot(true_xyz[:, 0], true_xyz[:, 1], true_xyz[:, 2],
        color='k', linewidth=2, alpha=0.8)

for scenario, group in df.sort_values(['PositionIndex', 'TrialIndex']).groupby('Scenario'):
    rgba = scenario_palette.get(scenario, (0.35, 0.35, 0.35, 1.0))
    ax.scatter(group['EstX'], group['EstY'], group['EstZ'],
               s=45, label=f'{scenario} estimates', alpha=0.65,
               color=rgba, depthshade=False)
    mean_track = group.groupby('PositionIndex')[['EstX', 'EstY', 'EstZ']].mean().reset_index().sort_values('PositionIndex')
    mean_xyz = mean_track[['EstX', 'EstY', 'EstZ']].values
    ax.plot(mean_xyz[:, 0], mean_xyz[:, 1], mean_xyz[:, 2],
            linestyle='--', linewidth=1.8, alpha=0.85, color=rgba)
    for _, row in group.iterrows():
        ax.plot([row['TrueX'], row['EstX']],
                [row['TrueY'], row['EstY']],
                [row['TrueZ'], row['EstZ']],
                color=rgba, alpha=0.25, linewidth=1.0)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Estimated vs True UE Positions in Environment')
ax.legend(loc='upper right', fontsize=8)

coord_min = np.min(np.vstack([
    true_xyz,
    df[['EstX', 'EstY', 'EstZ']].values
]), axis=0)
coord_max = np.max(np.vstack([
    true_xyz,
    df[['EstX', 'EstY', 'EstZ']].values
]), axis=0)
if mesh is not None:
    coord_min = np.minimum(coord_min, bounds[0])
    coord_max = np.maximum(coord_max, bounds[1])

ax.set_xlim(coord_min[0], coord_max[0])
ax.set_ylim(coord_min[1], coord_max[1])
ax.set_zlim(coord_min[2], coord_max[2])
ax.set_box_aspect(coord_max - coord_min)
ax.view_init(elev=22, azim=-45)
ax.grid(False)

fig.tight_layout()
fig.savefig(plot_dir / 'ue_positions_in_environment.png', dpi=200)
plt.close(fig)

# 10b. Top-down track overlay with point-cloud impressions
fig, ax = plt.subplots(figsize=(8, 6))
if mesh is not None:
    ax.scatter(mesh_verts[:, 0], mesh_verts[:, 1], s=0.3, alpha=0.01, color='gray')

ax.plot(true_xyz[:, 0], true_xyz[:, 1], color='k', linewidth=2, label='True path')
ax.scatter(true_xyz[:, 0], true_xyz[:, 1], c='k', s=55)

for scenario, group in df.groupby('Scenario'):
    rgba = scenario_palette.get(scenario, (0.35, 0.35, 0.35, 1.0))
    ax.scatter(group['EstX'], group['EstY'], label=f'{scenario} estimates', alpha=0.35, s=35, color=rgba)
    mean_track_2d = group.groupby('PositionIndex')[['EstX', 'EstY']].mean().reset_index().sort_values('PositionIndex')
    ax.plot(mean_track_2d['EstX'], mean_track_2d['EstY'], linestyle='--', linewidth=1.6, color=rgba, alpha=0.85)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Top-Down Tracks: True vs Estimated Clusters')
ax.legend(loc='upper right', fontsize=7)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(coord_min[0], coord_max[0])
ax.set_ylim(coord_min[1], coord_max[1])
ax.grid(True, linestyle=':', alpha=0.4)

fig.tight_layout()
fig.savefig(plot_dir / 'ue_tracks_topdown.png', dpi=200)
plt.close(fig)

# 11. Plotly interactive exports (if available)
if go is not None:
    inter_dir = ensure_directory(plot_dir / 'interactive')

    traces = []
    if mesh is not None:
        traces.append(go.Mesh3d(
            x=mesh_verts[:, 0], y=mesh_verts[:, 1], z=mesh_verts[:, 2],
            i=mesh_faces[:, 0], j=mesh_faces[:, 1], k=mesh_faces[:, 2],
            color='gray', opacity=0.08, name='Environment'
        ))

    tx_positions_for_plot = None
    if 'TxPositions' in df.columns:
        for entry in df['TxPositions']:
            arr = parse_matlab_matrix(entry)
            if arr.size and arr.ndim == 2 and arr.shape[1] >= 3:
                tx_positions_for_plot = arr[:, :3]
                break
    if tx_positions_for_plot is not None:
        traces.append(go.Scatter3d(
            x=tx_positions_for_plot[:, 0],
            y=tx_positions_for_plot[:, 1],
            z=tx_positions_for_plot[:, 2],
            mode='markers+text',
            text=[f'T{i+1}' for i in range(tx_positions_for_plot.shape[0])],
            textposition='top center',
            marker=dict(size=6, color='orange', symbol='diamond'),
            name='Transmitters'
        ))

    traces.append(go.Scatter3d(
        x=true_xyz[:, 0], y=true_xyz[:, 1], z=true_xyz[:, 2],
        mode='markers+lines', marker=dict(size=6, color='black'),
        line=dict(color='black', width=4), name='True UE path'
    ))

    err_min = float(df['HorizontalError'].min()) if df['HorizontalError'].notna().any() else 0.0
    err_max = float(df['HorizontalError'].max()) if df['HorizontalError'].notna().any() else 1.0

    scenario_groups = list(df.groupby('Scenario'))
    for group_idx, (scenario, group) in enumerate(scenario_groups):
        rgba = scenario_palette.get(scenario, (0.3, 0.3, 0.3, 1.0))
        color_hex = '#%02x%02x%02x' % tuple(int(255 * c) for c in rgba[:3])
        show_scale = (group_idx == 0)
        traces.append(go.Scatter3d(
            x=group['EstX'], y=group['EstY'], z=group['EstZ'],
            mode='markers',
            marker=dict(
                size=5,
                color=group['HorizontalError'],
                colorscale='Turbo',
                cmin=err_min,
                cmax=err_max,
                opacity=0.75,
                colorbar=dict(title='Horizontal Error (m)', x=-0.08, xanchor='left') if show_scale else None,
                showscale=show_scale
            ),
            name=f'{scenario} estimates',
            hovertemplate=(
                'Scenario: %{text}<br>'
                'Estimated: (%{x:.2f}, %{y:.2f}, %{z:.2f}) m<br>'
                'Horizontal error: %{marker.color:.2f} m<extra></extra>'
            ),
            text=[scenario] * len(group)
        ))
        mean_track = group.groupby('PositionIndex')[['EstX', 'EstY', 'EstZ']].mean().reset_index().sort_values('PositionIndex')
        traces.append(go.Scatter3d(
            x=mean_track['EstX'], y=mean_track['EstY'], z=mean_track['EstZ'],
            mode='lines', line=dict(color=color_hex, dash='dash', width=3),
            name=f'{scenario} mean track'
        ))

    layout3d = go.Layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        title='UE Path vs Estimates (Interactive 3D)',
        legend=dict(itemsizing='constant')
    )
    fig_interactive = go.Figure(data=traces, layout=layout3d)
    fig_interactive.write_html(inter_dir / 'ue_environment_3d.html', include_plotlyjs='cdn')

    if not gdop_valid.empty:
        gdop_traces = []
        for scenario, group in gdop_valid.groupby('Scenario'):
            rgba = scenario_palette.get(scenario, (0.3, 0.3, 0.3, 1.0))
            color_hex = '#%02x%02x%02x' % tuple(int(255 * c) for c in rgba[:3])
            gdop_traces.append(go.Scatter(
                x=group['GDOP_Selected'],
                y=group['HorizontalError'],
                mode='markers',
                marker=dict(size=7, color=color_hex, opacity=0.7),
                name=scenario,
                hovertemplate=(
                    'Scenario: %{text}<br>'
                    'GDOP: %{x:.2f}<br>'
                    'Horizontal error: %{y:.2f} m<br>'
                    'Position index: %{customdata[0]}<br>'
                    'Trial index: %{customdata[1]}<extra></extra>'
                ),
                text=[scenario] * len(group),
                customdata=np.column_stack((
                    group['PositionIndex'].to_numpy(),
                    group['TrialIndex'].to_numpy()
                ))
            ))
        title_suffix = f' (Pearson r={gdop_pearson:.2f})' if np.isfinite(gdop_pearson) else ''
        gdop_layout = go.Layout(
            xaxis=dict(title='GDOP (Selected Anchors)'),
            yaxis=dict(title='Horizontal Error (m)'),
            title='GDOP vs Horizontal Error' + title_suffix,
            legend=dict(itemsizing='constant')
        )
        fig_gdop = go.Figure(data=gdop_traces, layout=gdop_layout)
        fig_gdop.write_html(inter_dir / 'gdop_vs_error.html', include_plotlyjs='cdn')

    topdown_traces = []
    if mesh is not None:
        topdown_traces.append(go.Scattergl(
            x=mesh_verts[:, 0], y=mesh_verts[:, 1],
            mode='markers', marker=dict(size=1, color='gray', opacity=0.05),
            name='Environment footprint'
        ))

    topdown_traces.append(go.Scatter(
        x=true_xyz[:, 0], y=true_xyz[:, 1],
        mode='markers+lines', marker=dict(size=7, color='black'),
        line=dict(color='black', width=4), name='True path'
    ))

    for scenario, group in df.groupby('Scenario'):
        rgba = scenario_palette.get(scenario, (0.3, 0.3, 0.3, 1.0))
        color_hex = '#%02x%02x%02x' % tuple(int(255 * c) for c in rgba[:3])
        topdown_traces.append(go.Scatter(
            x=group['EstX'], y=group['EstY'],
            mode='markers', marker=dict(size=6, color=color_hex, opacity=0.6),
            name=f'{scenario} estimates'
        ))

    layout2d = go.Layout(
        xaxis=dict(title='X (m)', scaleanchor='y'),
        yaxis=dict(title='Y (m)'),
        title='Top-Down UE Tracks (Interactive)',
        legend=dict(itemsizing='constant')
    )
    fig_topdown = go.Figure(data=topdown_traces, layout=layout2d)
    fig_topdown.write_html(inter_dir / 'ue_tracks_topdown.html', include_plotlyjs='cdn')
else:
    print('Plotly not available; skipping interactive exports. Install with "pip install plotly" to enable.')

print('Plots saved to', plot_dir)
