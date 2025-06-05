# database/db_manager.py - REFACTORED FULL VERSION
import os
import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import logging
import traceback

print("⚠️ SQLAlchemy disabled - using psycopg2 only")

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.database_url = os.getenv('DATABASE_URL')
        self.logger = logging.getLogger(__name__)

        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")

        self._connect()
        self._create_tables()

    def _connect(self):
        try:
            self.connection = psycopg2.connect(self.database_url)
            self.connection.autocommit = True
            print("✅ Connected to PostgreSQL (psycopg2 only)")
        except Exception as e:
            print(f"❌ DB connection failed: {e}")
            traceback.print_exc()
            raise

    def _create_tables(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMPTZ NOT NULL,
                        input_token VARCHAR(10) NOT NULL,
                        output_token VARCHAR(10) NOT NULL,
                        amount_in DECIMAL(18,8) NOT NULL,
                        amount_out DECIMAL(18,8) NOT NULL,
                        price_impact DECIMAL(10,6),
                        price DECIMAL(18,8),
                        volume DECIMAL(18,8),
                        rsi DECIMAL(5,2),
                        profitable BOOLEAN,
                        ml_prediction JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ml_models (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(100) NOT NULL,
                        model_type VARCHAR(50) NOT NULL,
                        performance_metrics JSONB,
                        training_samples INTEGER,
                        accuracy DECIMAL(5,2),
                        r2_score DECIMAL(8,4),
                        mae DECIMAL(10,6),
                        model_file_path TEXT,
                        is_active BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        last_trained TIMESTAMPTZ
                    );
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_transactions_profitable ON transactions(profitable);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_models_active ON ml_models(is_active);")
            print("✅ Tables ensured")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            traceback.print_exc()

    def _convert_numpy_values(self, value):
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, (np.bool_, bool)):
            return bool(value)
        return value

    def save_transaction(self, trade_data: Dict, market_data: Optional[Dict] = None, ml_prediction: Optional[Dict] = None):
        try:
            with self.connection.cursor() as cursor:
                timestamp = pd.to_datetime(trade_data.get('timestamp', datetime.utcnow()))
                input_token = trade_data.get('input_token', 'SOL')
                output_token = trade_data.get('output_token', 'USDC')
                amount_in = float(trade_data.get('amount_in', 0))
                amount_out = float(trade_data.get('amount_out', 0))
                price_impact = float(trade_data.get('price_impact', 0))
                price = float(market_data.get('price', 0)) if market_data else 0
                rsi = float(market_data.get('rsi', 50)) if market_data else 50
                volume = amount_in
                profitable = amount_out > amount_in

                cursor.execute("""
                    INSERT INTO transactions (timestamp, input_token, output_token, amount_in, amount_out,
                        price_impact, price, volume, rsi, profitable, ml_prediction)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    timestamp, input_token, output_token, amount_in, amount_out,
                    price_impact, price, volume, rsi, profitable,
                    psycopg2.extras.Json(ml_prediction) if ml_prediction else None
                ))
                transaction_id = cursor.fetchone()[0]
                print(f"✅ Transaction saved (ID: {transaction_id})")
                return transaction_id
        except Exception as e:
            print(f"❌ Error saving transaction: {e}")
            traceback.print_exc()
            return None

    def get_recent_transactions(self, limit: int = 100) -> pd.DataFrame:
        try:
            query = """
                SELECT timestamp, input_token, output_token, amount_in, amount_out,
                       price_impact, price, volume, rsi, profitable, ml_prediction
                FROM transactions
                ORDER BY timestamp DESC
                LIMIT %s;
            """
            df = pd.read_sql_query(query, self.connection, params=(limit,))
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"✅ Retrieved {len(df)} recent transactions")
            else:
                print("⚠️ No recent transactions found")
            return df
        except Exception as e:
            print(f"❌ Error retrieving recent transactions: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def get_all_transactions_for_ml(self) -> pd.DataFrame:
        try:
            query = """
                SELECT timestamp, price, volume, rsi, amount_in, amount_out,
                       price_impact, profitable, input_token, output_token
                FROM transactions
                WHERE price IS NOT NULL AND price > 0
                  AND rsi IS NOT NULL AND rsi BETWEEN 1 AND 99
                  AND volume IS NOT NULL AND volume > 0
                ORDER BY timestamp ASC;
            """
            df = pd.read_sql_query(query, self.connection)
            if df.empty:
                print("⚠️ No ML data found")
                return df
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            before = len(df)
            df = df[
                (df['price'] > 0.01) &
                (df['rsi'].between(1, 99)) &
                (df['amount_in'] > 0) &
                (df['amount_out'] > 0)
            ]
            after = len(df)
            print(f"✅ ML data: {after}/{before} records retained")
            return df
        except Exception as e:
            print(f"❌ Error loading ML data: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def save_ml_model_info(self, model_info: Dict):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    UPDATE ml_models SET is_active = FALSE WHERE model_type = %s;
                """, (model_info.get('model_type', 'unknown'),))

                metrics = model_info.get('metrics', {})
                converted_metrics = {k: self._convert_numpy_values(v) for k, v in metrics.items()}

                cursor.execute("""
                    INSERT INTO ml_models (model_name, model_type, performance_metrics, training_samples,
                        accuracy, r2_score, mae, model_file_path, is_active, last_trained)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """, (
                    model_info.get('model_name', 'Unknown'),
                    model_info.get('model_type', 'unknown'),
                    psycopg2.extras.Json(converted_metrics),
                    int(model_info.get('training_samples', 0)),
                    self._convert_numpy_values(model_info.get('accuracy', 0)),
                    self._convert_numpy_values(model_info.get('r2_score', 0)),
                    self._convert_numpy_values(model_info.get('mae', 0)),
                    model_info.get('model_file_path', ''),
                    True,
                    datetime.utcnow()
                ))
                model_id = cursor.fetchone()[0]
                print(f"✅ ML model info saved (ID: {model_id})")
                return model_id
        except Exception as e:
            print(f"❌ Error saving ML model info: {e}")
            traceback.print_exc()
            return None

    def get_ml_model_performance(self) -> Dict:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT model_name, model_type, accuracy, r2_score, mae,
                           training_samples, last_trained
                    FROM ml_models
                    WHERE is_active = TRUE
                    ORDER BY last_trained DESC;
                """)
                results = cursor.fetchall()
            performance = {}
            for row in results:
                model_name = row[0]
                performance[model_name] = {
                    'model_type': row[1],
                    'accuracy': float(row[2]) if row[2] else 0,
                    'r2': float(row[3]) if row[3] else 0,
                    'mae': float(row[4]) if row[4] else 0,
                    'training_samples': row[5] or 0,
                    'last_trained': row[6].strftime('%Y-%m-%d %H:%M:%S') if row[6] else 'Never'
                }
            return performance
        except Exception as e:
            print(f"❌ Error getting ML performance: {e}")
            traceback.print_exc()
            return {}

    def get_transaction_count(self) -> int:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM transactions;")
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"❌ Error getting transaction count: {e}")
            return 0

    def get_database_stats(self) -> Dict:
        try:
            stats = {}
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM transactions;")
                stats['total_transactions'] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM transactions WHERE profitable = TRUE;")
                stats['profitable_transactions'] = cursor.fetchone()[0]
                cursor.execute("SELECT AVG(amount_out - amount_in) FROM transactions;")
                avg_pnl = cursor.fetchone()[0]
                stats['avg_pnl'] = float(avg_pnl) if avg_pnl else 0
                cursor.execute("SELECT COUNT(*) FROM ml_models WHERE is_active = TRUE;")
                stats['active_ml_models'] = cursor.fetchone()[0]
            return stats
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {}

    def migrate_from_csv(self, csv_file_path: str) -> bool:
        try:
            if not os.path.exists(csv_file_path):
                print(f"⚠️ CSV file not found: {csv_file_path}")
                return False
            df = pd.read_csv(csv_file_path)
            print(f"📊 Found {len(df)} records in CSV")
            if df.empty:
                print("⚠️ CSV file is empty")
                return False
            if self.get_transaction_count() > 0:
                print("⚠️ Database already contains transactions. Skipping migration.")
                return True
            migrated = 0
            for _, row in df.iterrows():
                try:
                    trade_data = {
                        'timestamp': row.get('timestamp'),
                        'input_token': row.get('input_token', 'SOL'),
                        'output_token': row.get('output_token', 'USDC'),
                        'amount_in': row.get('amount_in', 0),
                        'amount_out': row.get('amount_out', 0),
                        'price_impact': row.get('price_impact', 0)
                    }
                    market_data = {
                        'price': row.get('price', 0),
                        'rsi': row.get('rsi', 50)
                    }
                    if self.save_transaction(trade_data, market_data):
                        migrated += 1
                except Exception as e:
                    print(f"⚠️ Error migrating row: {e}")
                    continue
            print(f"✅ Migrated {migrated}/{len(df)} rows from CSV")
            return True
        except Exception as e:
            print(f"❌ Migration from CSV failed: {e}")
            traceback.print_exc()
            return False

    def close(self):
        if self.connection:
            self.connection.close()
            print("✅ PostgreSQL connection closed")

_db_manager = None

def get_db_manager():
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
