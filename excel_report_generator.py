"""
excel_report_generator.py

仮想通貨バックテスト結果を基に、収益性評価のための
自動Excelレポートを生成するモジュール（構想最上位レベル準拠）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image as ExcelImage
from tempfile import NamedTemporaryFile


class ExcelReportGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.wb = Workbook()
        self.wb.remove(self.wb.active)  # Remove default sheet

    def add_summary_sheet(self):
        sheet = self.wb.create_sheet("サマリー")
        df = self.df.copy()
        df["result"] = np.where(df["profit"] > 0, "win", "lose")

        grouped = df.groupby(["symbol", "position"])
        summary = grouped.agg(
            win_rate=("result", lambda x: float(np.mean(x == "win"))),  # float()でスカラー値に変換
            pf=("profit", lambda x: float(x[x > 0].sum() / abs(x[x < 0].sum()) if any(x < 0) and x[x > 0].sum() > 0 else 0)),  # ゼロ除算対策とfloat変換
            trade_count=("profit", "count"),
            avg_profit=("profit", "mean"),
            max_dd=("profit", lambda x: float((x.cumsum() - x.cumsum().expanding().max()).min())),  # max_ddの計算を修正
        ).reset_index()

        for row in dataframe_to_rows(summary, index=False, header=True):
            sheet.append(row)

    def add_charts_sheet(self):
        sheet = self.wb.create_sheet("Charts")
        df = self.df.copy()

        def plot_and_insert(x_col, y_col, anchor):
            data = df[[x_col, y_col]].dropna()
            if data.empty:
                return
            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                plt.figure(figsize=(6, 4))
                sns.scatterplot(x=x_col, y=y_col, data=data, alpha=0.5)
                sns.kdeplot(data=data, x=x_col, y=y_col, fill=True, alpha=0.3, thresh=0.1)
                plt.title(f"{x_col} vs {y_col}")
                plt.tight_layout()
                plt.savefig(tmpfile.name)
                plt.close()
                img = ExcelImage(tmpfile.name)
                sheet.add_image(img, anchor)

        anchors = ["B2", "B22", "B42", "B62", "B82", "B102", "B122"]
        all_score_cols = [col for col in df.columns if col.endswith("_long") or col.endswith("_short")]
        plotted = 0

        for col in all_score_cols:
            if col in df.columns and plotted < len(anchors):
                plot_and_insert(col, "profit", anchors[plotted])
                plotted += 1

        # 表としても出力
        summary_sheet = self.wb.create_sheet("Score_vs_Profit")
        summary_data = []
        for symbol in df["symbol"].dropna().unique():
            df_symbol = df[df["symbol"] == symbol]
            for col in all_score_cols:
                if col in df_symbol.columns:
                    sub_df = df_symbol[[col, "profit"]].dropna()
                    try:
                        bins = pd.qcut(sub_df[col], q=10, duplicates='drop')
                    except ValueError:
                        continue
                    grouped = sub_df.groupby(bins).agg(
                        avg_score=(col, "mean"),
                        avg_profit=("profit", "mean"),
                        win_rate=("profit", lambda x: np.mean(x > 0)),
                        count=("profit", "count")
                    ).reset_index()
                    grouped.insert(0, "symbol", symbol)
                    grouped.insert(1, "score_feature", col)
                    summary_data.append(grouped)

        if summary_data:
            final_df = pd.concat(summary_data, ignore_index=True)

            def safe_convert(val):
                if isinstance(val, pd.Interval):
                    return str(val)
                return val

            for i, row in enumerate(dataframe_to_rows(final_df, index=False, header=True)):
                clean_row = [safe_convert(cell) for cell in row]
                if len(clean_row) >= 7:
                    rearranged = clean_row[:2] + clean_row[3:7] + [clean_row[2]] + clean_row[7:]
                    summary_sheet.append(rearranged)
                else:
                    summary_sheet.append(clean_row)

            # 条件付き書式の追加（勝率・平均利益列）
            from openpyxl.formatting.rule import ColorScaleRule

            # 勝率 → D列（=列番号4）
            rule_win = ColorScaleRule(start_type='percentile', start_value=0, start_color='FFAAAA',
                                      mid_type='percentile', mid_value=50, mid_color='FFFFAA',
                                      end_type='percentile', end_value=100, end_color='AAFFAA')
            summary_sheet.conditional_formatting.add(f'D2:D{summary_sheet.max_row}', rule_win)

            # 平均利益 → E列（=列番号5）
            rule_profit = ColorScaleRule(start_type='percentile', start_value=0, start_color='FFAAAA',
                                         mid_type='percentile', mid_value=50, mid_color='FFFFAA',
                                         end_type='percentile', end_value=100, end_color='AAFFAA')
            summary_sheet.conditional_formatting.add(f'E2:E{summary_sheet.max_row}', rule_profit)


    def add_score_analysis_sheet(self):
        sheet = self.wb.create_sheet("スコア分析")
        df = self.df.copy()

        score_cols = []
        if "buy_score" in df.columns:
            score_cols.append("buy_score")
        if "sell_score" in df.columns:
            score_cols.append("sell_score")

        if not score_cols:
            sheet.append(["スコア列が見つかりません"])
            return

        try:
            for symbol in df["symbol"].dropna().unique():
                df_symbol = df[df["symbol"] == symbol]
                for score_col in score_cols:
                    valid_data = df_symbol[df_symbol[score_col].notna()].copy()
                    if len(valid_data) == 0:
                        continue

                    def get_score_bin(score):
                        if pd.isna(score):
                            return "N/A"
                        elif score < 0.1:
                            return "0.0-0.1"
                        elif score < 0.2:
                            return "0.1-0.2"
                        elif score < 0.3:
                            return "0.2-0.3"
                        elif score < 0.4:
                            return "0.3-0.4"
                        elif score < 0.5:
                            return "0.4-0.5"
                        elif score < 0.6:
                            return "0.5-0.6"
                        elif score < 0.7:
                            return "0.6-0.7"
                        elif score < 0.8:
                            return "0.7-0.8"
                        elif score < 0.9:
                            return "0.8-0.9"
                        else:
                            return "0.9-1.0"

                    valid_data["score_bin"] = valid_data[score_col].apply(get_score_bin)

                    analysis = valid_data.groupby("score_bin").agg(
                        勝率=("profit", lambda x: float(np.mean(x > 0))),
                        平均利益=("profit", "mean"),
                        取引回数=("profit", "count"),
                        平均スコア=(score_col, "mean")
                    ).reset_index()
                    analysis.insert(0, "スコア種別", score_col)
                    analysis.insert(0, "通貨ペア", symbol)

                    if sheet.max_row > 1:
                        sheet.append([])  # 空行で区切る

                    for row in dataframe_to_rows(analysis, index=False, header=True):
                        sheet.append(row)

        except Exception as e:
            sheet.append([f"スコア分析でエラー: {str(e)}"])

    def add_market_condition_sheet(self):
        sheet = self.wb.create_sheet("市場状況")
        df = self.df.copy()

        required_cols = ["ADX", "ATR", "RSI", "symbol"]
        if all(col in df.columns for col in required_cols):
            try:
                df = df.dropna(subset=required_cols + ["profit"]).copy()

                # 各指標を5段階にビニング
                adx_labels = ["very_low", "low", "medium", "high", "very_high"]
                atr_labels = ["very_low", "low", "medium", "high", "very_high"]
                rsi_labels = ["very_low", "low", "medium", "high", "very_high"]

                df["ADX_bin"] = pd.qcut(df["ADX"], q=5, labels=adx_labels)
                df["ATR_bin"] = pd.qcut(df["ATR"], q=5, labels=atr_labels)
                df["RSI_bin"] = pd.qcut(df["RSI"], q=5, labels=rsi_labels)

                # グルーピング
                grouped = df.groupby(["symbol", "ADX_bin", "ATR_bin", "RSI_bin"]).agg(
                    win_rate=("profit", lambda x: float(np.mean(x > 0))),
                    avg_profit=("profit", "mean"),
                    count=("profit", "count")
                ).reset_index()

                for row in dataframe_to_rows(grouped, index=False, header=True):
                    sheet.append(row)

            except Exception as e:
                sheet.append([f"市場状況分析でエラー: {str(e)}"])
        else:
            sheet.append(["ADX, ATR, RSI, または symbol が見つかりません"])


    def add_symbol_sheets(self):
        for symbol, sdf in self.df.groupby("symbol"):
            sheet = self.wb.create_sheet(f"{symbol}")
            sdf = sdf.copy()
            sdf["result"] = np.where(sdf["profit"] > 0, "win", "lose")

            summary = sdf.groupby("position").agg(
                win_rate=("result", lambda x: np.mean(x == "win")),
                avg_profit=("profit", "mean"),
                count=("profit", "count"),
                avg_duration=("holding_period", "mean") if "holding_period" in sdf.columns else ("profit", "count")
            ).reset_index()

            sheet.append(["ポジション別の成績"])
            for row in dataframe_to_rows(summary, index=False, header=True):
                sheet.append(row)

            # 勝ち/負け特徴比較（拡張版）
            sheet.append([])
            sheet.append(["勝ち/負けトレードの特徴比較"])
            features = ["buy_score", "sell_score", "ATR", "ADX"]
            if "holding_period" in sdf.columns:
                features.append("holding_period")

            available_features = [col for col in features if col in sdf.columns]
            if available_features:
                compare = sdf.groupby("result")[available_features].mean().reset_index()
                for row in dataframe_to_rows(compare, index=False, header=True):
                    sheet.append(row)

            sheet.append([])
            sheet.append(["トレード一覧"])
            for row in dataframe_to_rows(sdf.head(50), index=False, header=True):
                sheet.append(row)


    def save(self, path: str):
        self.wb.save(path)
