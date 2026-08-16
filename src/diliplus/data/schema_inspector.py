"""
DILI-PLUS | 关键物理表结构检查工具（包实现）

职责：只读检查实验所需的化验、用药、病历和诊断表，并打印少量字段样本。
输入：D:\MedicalAI_Work\duck\medical.duckdb。
输出：仅输出到终端，不生成建模数据。
状态：数据库字段映射的人工核查工具。
安全：数据库连接显式使用 read_only=True。
"""

from diliplus.config import load_settings
from diliplus.database import connect_source_database

def inspect_physical_schema(settings=None):
    # 从集中配置读取跨域 DuckDB 路径
    settings = settings or load_settings()
    db_path = str(settings.database_path)
    
    print(f"🔗 [DILIPLUS] Connecting to database: {db_path}\n")
    # 强制只读模式
    conn = connect_source_database(settings)
    
    tables_to_inspect = [
        'laboratory_report', 
        'laboratory_report_sub',
        'medication_order',    
        'medical_record_home', 
        'inpatient_diagnosis'  
    ]
    
    existing_tables_df = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").df()
    existing_tables = existing_tables_df['table_name'].tolist()
    
    valid_tables = [t for t in tables_to_inspect if t in existing_tables]
    
    if not valid_tables:
        print("⚠️ [DILIPLUS] Target tables not found. Defaulting to basic lab tables.")
        valid_tables = [t for t in ['laboratory_report', 'laboratory_report_sub'] if t in existing_tables]

    for table in valid_tables:
        print("=" * 60)
        print(f" 📂 TARGET TABLE: {table.upper()}")
        print("=" * 60)
        
        schema_df = conn.execute(f"DESCRIBE {table}").df()
        columns = schema_df['column_name'].tolist()
        print(f"📝 [Physical Columns] ({len(columns)} total):")
        for i in range(0, len(columns), 5):
            print(" | ".join(columns[i:i+5]))
            
        filter_col = next((c for c in columns if any(k in c for k in ['class_name', 'result_valu', 'time', 'date', 'code'])), columns[0])
        
        print(f"\n🔍 [Data Peeking] (First 3 non-null rows based on '{filter_col}'):")
        try:
            sample_query = f"SELECT * FROM {table} WHERE {filter_col} IS NOT NULL LIMIT 3"
            sample_df = conn.execute(sample_query).df()
            
            key_keywords = ['report', 'event', 'inpatient', 'class', 'result', 'abnormal', 'time', 'date', 'code', 'drug', 'dose']
            selected_cols = [c for c in columns if any(k in c.lower() for k in key_keywords)]
            
            if not selected_cols:
                selected_cols = columns[:8] 
                
            print(sample_df[selected_cols].to_markdown(index=False))
        except Exception as e:
            print(f"🚨 Failed to fetch sample data: {e}")
        print("\n")

    conn.close()

if __name__ == "__main__":
    inspect_physical_schema()
