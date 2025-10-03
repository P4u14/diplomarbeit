import warnings
from matplotlib import pyplot as plt
import numpy as np

from visualization.base_plotter import BasePlotter


class MetricCountBarPlotter(BasePlotter):
    def __init__(self, experiments, metrics, directory='metric_count_bar_plots'):
        super().__init__(experiments, metrics, directory)

    def plot(self, data_frames, output_dir):
        # expect a single experiment's full data ('_all.csv')
        if not data_frames:
            warnings.warn('No data to plot.')
            return
        df = data_frames[0]
        # Farben: Blau und Grün
        color_list = ['#1f77b4', '#2ca02c']
        n_metrics = len(self.metrics)
        colors = lambda i: color_list[i % 2]
        # prepare sorted unique metric values (über alle Metriken)
        all_vals = []
        for metric in self.metrics:
            if metric in df.columns:
                all_vals.extend(df[metric].dropna().unique())
        sorted_vals = sorted(set(all_vals))
        x = np.arange(len(sorted_vals))
        bar_width = 0.8 / n_metrics
        # 1) prozentuale counts (gruppiert)
        fig1, ax1 = plt.subplots(figsize=(max(3, int(len(sorted_vals) * 0.3)), 6))
        bars1 = []
        total_count = len(df)
        for i, metric in enumerate(self.metrics):
            if metric not in df.columns:
                continue
            vals_all = df[metric].dropna()
            counts_abs = vals_all.value_counts().reindex(sorted_vals, fill_value=0)
            percent = (counts_abs.values / total_count) * 100 if total_count > 0 else np.zeros_like(counts_abs.values)
            bars = ax1.bar(x + i*bar_width, percent, width=bar_width, label=metric, color=colors(i))
            bars1.extend(bars)
            for bar, p in zip(bars, percent):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{p:.1f}", ha='center', va='bottom')
        ax1.set_title(f"Prozentualer Anteil der Bilder pro Metrikwert")
        ax1.set_xlabel('Metric Value')
        ax1.set_ylabel('Prozent')
        ax1.set_xticks(x + 0.4)
        ax1.set_xticklabels([str(v) for v in sorted_vals], rotation=45, ha='right')
        ax1.set_ylim(0, 100)
        ax1.legend(ncol=n_metrics, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', frameon=False)
        fig1.tight_layout()
        fig1.subplots_adjust(bottom=0.3)
        fn1 = f"grouped_percent_bar.png"
        self.save_plot(fig1, output_dir, filename=fn1)
        # 2) counts per dataset (gruppiert)
        if 'Dataset' in df.columns:
            ds_groups = [d for d in df['Dataset'].unique() if d not in ('All Datasets','Healthy','Sick')]
            if ds_groups:
                total_w = 0.8
                n_ds = len(ds_groups)
                fig2, ax2 = plt.subplots(figsize=(max(3, int(len(sorted_vals)*0.3)), 6))
                for j, metric in enumerate(self.metrics):
                    if metric not in df.columns:
                        continue
                    for i, grp in enumerate(ds_groups):
                        group_rows = df[df['Dataset'] == grp]
                        num_imgs = len(group_rows)
                        vals_grp = group_rows[metric].dropna()
                        cnt = vals_grp.value_counts().reindex(sorted_vals, fill_value=0)
                        pos = x + (j * n_ds + i) * (bar_width / n_ds)
                        bars = ax2.bar(pos, cnt.values, width=bar_width / n_ds, label=f"{metric} - {grp} ({num_imgs})", color=colors(j))
                        for bar in bars:
                            h = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2, h, str(int(h)), ha='center', va='bottom')
                ax2.set_title(f"Count by metric value and dataset")
                ax2.set_xlabel('Metric Value'); ax2.set_ylabel('Count')
                ax2.set_xticks(x + total_w/2 - bar_width/2)
                ax2.set_xticklabels([str(v) for v in sorted_vals], rotation=45, ha='right')
                ax2.legend(ncol=min(len(ds_groups)*n_metrics, 3), loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', frameon=False)
                fig2.tight_layout()
                fig2.subplots_adjust(bottom=0.3)
                fn2 = f"grouped_dataset_count_bar.png"
                self.save_plot(fig2, output_dir, filename=fn2)
        # 3) counts per health status (gruppiert)
        if 'Sick' in df.columns:
            status = [('Healthy',0),('Sick',1)]
            n_status = len(status)
            fig3, ax3 = plt.subplots(figsize=(max(3, int(len(sorted_vals)*0.3)), 6))
            for j, metric in enumerate(self.metrics):
                if metric not in df.columns:
                    continue
                for i, (name, flag) in enumerate(status):
                    group_rows = df[df['Sick'] == flag]
                    num_imgs = len(group_rows)
                    vals_st = group_rows[metric].dropna()
                    cnt = vals_st.value_counts().reindex(sorted_vals, fill_value=0)
                    pos = x + (j * n_status + i) * (bar_width / n_status)
                    bars = ax3.bar(pos, cnt.values, width=bar_width / n_status, label=f"{metric} - {name} ({num_imgs})", color=colors(j))
                    for bar in bars:
                        h = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2, h, str(int(h)), ha='center', va='bottom')
            ax3.set_title(f"Count by metric value and health status")
            ax3.set_xlabel('Metric Value'); ax3.set_ylabel('Count')
            ax3.set_xticks(x + 0.4)
            ax3.set_xticklabels([str(v) for v in sorted_vals], rotation=45, ha='right')
            ax3.legend(ncol=min(n_status*n_metrics, 3), loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', frameon=False)
            fig3.tight_layout()
            fig3.subplots_adjust(bottom=0.3)
            fn3 = f"grouped_status_count_bar.png"
            self.save_plot(fig3, output_dir, filename=fn3)
        # 4) histogram of metric distribution (überlagert)
        fig4, ax4 = plt.subplots(figsize=(max(3, int(len(sorted_vals) * 0.3)), 6))
        for i, metric in enumerate(self.metrics):
            if metric not in df.columns:
                continue
            vals_all = df[metric].dropna()
            ax4.hist(vals_all, bins='auto', color=colors(i), alpha=0.5, label=metric)
        ax4.set_title(f"Histogram of metric values")
        ax4.set_xlabel('Metric Value')
        ax4.set_ylabel('Frequency')
        ax4.legend(ncol=n_metrics, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize='small', frameon=False)
        fig4.tight_layout()
        fig4.subplots_adjust(bottom=0.3)
        fn4 = f"grouped_histogram.png"
        self.save_plot(fig4, output_dir, filename=fn4)
