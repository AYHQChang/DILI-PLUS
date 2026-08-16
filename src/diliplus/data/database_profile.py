"""
DILI-PLUS | 数据资产概览工具（包实现）

职责：只读扫描外部医疗 DuckDB 的主 schema，统计表规模与字段结构。
输入：D:\MedicalAI_Work\duck\medical.duckdb。
输出：reports/00_DILIPLUS_Global_Asset_Report.md。
状态：主流程前置诊断工具；不参与队列构建或模型训练。
安全：数据库连接显式使用 read_only=True。
"""

import os
import pandas as pd
from datetime import datetime

from diliplus.config import load_settings
from diliplus.database import connect_source_database

def generate_global_database_profile(settings=None):
    # 1. 严格的项目目录定位
    settings = settings or load_settings()
    report_dir = str(settings.paths.reports)
    os.makedirs(report_dir, exist_ok=True)
    
    # 2. 从集中配置读取全局 DuckDB 路径（连接工厂固定为只读）
    db_path = str(settings.database_path)
    report_path = os.path.join(report_dir, "00_DILIPLUS_Global_Asset_Report.md")
    
    print(f"🔗 [DILIPLUS] Connecting to database at: {db_path}")
    
    try:
        # 强制只读模式，确保底层数据零污染
        conn = connect_source_database(settings)
    except Exception as e:
        print(f"🚨 [DILIPLUS] Failed to connect to DuckDB: {e}")
        return

    # 3. 获取所有表名 (过滤掉系统内置表)
    tables_df = conn.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
    """).df()
    tables = tables_df['table_name'].tolist()
    
    print(f"🔍 [DILIPLUS] Found {len(tables)} tables. Profiling row counts and schemas for dual-stream extraction...")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 🏥 DILIPLUS Global Asset Report\n\n")
        f.write(f"> **Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"> **Database Path**: `{db_path}` (Read-Only Mode)\n")
        f.write(f"> **Objective**: Foundation for Bounded Right-Censoring and Dual-Stream Polypharmacy Architecture.\n\n")
        
        # 4. 统计全局表级宏观数据
        summary_data = []
        for t in tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                summary_data.append({"Table Name": t, "Row Count": count})
            except Exception as e:
                summary_data.append({"Table Name": t, "Row Count": "Error/Inaccessible"})
        
        summary_df = pd.DataFrame(summary_data)
        summary_df['Sort_Key'] = pd.to_numeric(summary_df['Row Count'], errors='coerce').fillna(-1)
        summary_df = summary_df.sort_values(by="Sort_Key", ascending=False).drop(columns=['Sort_Key'])
        
        f.write("## 1. 📊 Database Summary (Sorted by Volume)\n\n")
        f.write(summary_df.to_markdown(index=False) + "\n\n")
        
        # 5. 逐表深入探查 Schema
        f.write("## 2. 📋 Detailed Table Schemas\n\n")
        for t in tables:
            f.write(f"### Table: `{t}`\n")
            try:
                schema_df = conn.execute(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{t}'
                    ORDER BY ordinal_position
                """).df()
                f.write(schema_df.to_markdown(index=False) + "\n\n")
            except Exception as e:
                f.write(f"> ⚠️ Error profiling schema for {t}: {e}\n\n")

    conn.close()
    print(f"✅ [DILIPLUS] Profiling complete. Report generated at: {report_path}")

if __name__ == "__main__":
    generate_global_database_profile()
