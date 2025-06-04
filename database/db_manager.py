# database/db_manager.py - PostgreSQL Integration for DexBot (FIXED CRITICAL BUG)
import os
import psycopg2
import psycopg2.extras  # DODANE: Potrzebne dla Json()
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

# DODANE: SQLAlchemy imports
try:
    import sqlalchemy
    from sqlalchemy import create_engine
    SQLALCHEMY_AVAILABLE = True
    print("✅ SQLAlchemy available for enhanced ML pipeline")
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("⚠️ SQLAlchemy not available - using psycopg2 only")


class DatabaseManager:
    """
    PostgreSQL database manager with enhanced ML pipeline support
    """

    def __init__(self):
        self.connection = None
        self.database_url = os.getenv('DATABASE_URL')
        self.logger = logging.getLogger(__name__)
        self.sqlalchemy_engine = None

        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")

        self._connect()
        self._create_tables()
        
        # Initialize SQLAlchemy engine for better pandas support
        if SQLALCHEMY_AVAILABLE:
            self._init_sqlalchemy_engine()

    def _connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(self.database_url)
            self.connection.autocommit = True
            print("✅ Connected to PostgreSQL database")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise

    def _init_sqlalchemy_engine(self):
        """Initialize SQLAlchemy engine for pandas compatibility"""
        try:
            if self.database_url:
                # Create SQLAlchemy engine from DATABASE_URL
                self.sqlalchemy_engine = create_engine(self.database_url)
                print("✅ SQLAlchemy engine initialized for ML pipeline")
        except Exception as e:
            print(f"⚠️ SQLAlchemy engine init failed: {e}")
            self.sqlalchemy_engine = None

    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            cursor = self.connection.cursor()

            # Transactions table
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

            # ML Models table
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

            # Create indexes for better performance
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_transactions_profitable ON transactions(profitable);")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ml_models_active ON ml_models(is_active);")

            cursor.close()
            print("✅ Database tables created successfully")

        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            raise

    def save_transaction(self, trade_data: Dict,
                         market_data: Optional[Dict] = None, ml_prediction: Optional[Dict] = None):
        """Save transaction to database"""
        try:
            cursor = self.connection.cursor()

            # Extract data with fallbacks
            timestamp = trade_data.get(
                'timestamp', datetime.utcnow().isoformat())
            input_token = trade_data.get('input_token', 'SOL')
            output_token = trade_data.get('output_token', 'USDC')
            amount_in = float(trade_data.get('amount_in', 0))
            amount_out = float(trade_data.get('amount_out', 0))
            price_impact = float(trade_data.get('price_impact', 0))

            # Market data
            price = float(market_data.get('price', 0)) if market_data else 0
            volume = float(amount_in)  # Use transaction amount as volume
            rsi = float(market_data.get('rsi', 50)) if market_data else 50

            # Calculate profitability
            profitable = amount_out > amount_in

            # ML prediction as JSON
            ml_pred_json = ml_prediction if ml_prediction else None

            cursor.execute("""
                INSERT INTO transactions
                (timestamp, input_token, output_token, amount_in, amount_out,
                 price_impact, price, volume, rsi, profitable, ml_prediction)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                timestamp, input_token, output_token, amount_in, amount_out,
                price_impact, price, volume, rsi, profitable,
                psycopg2.extras.Json(ml_pred_json) if ml_pred_json else None
            ))

            transaction_id = cursor.fetchone()[0]
            cursor.close()

            print(f"✅ Transaction saved to database (ID: {transaction_id})")
            return transaction_id

        except Exception as e:
            print(f"❌ Error saving transaction: {e}")
            return None

    def get_recent_transactions(self, limit: int = 100) -> pd.DataFrame:
        """ENHANCED: Get recent transactions with SQLAlchemy support"""
        try:
            query = """
                SELECT timestamp, input_token, output_token, amount_in, amount_out,
                       price_impact, price, volume, rsi, profitable, ml_prediction
                FROM transactions
                ORDER BY timestamp DESC
                LIMIT %s;
            """
        
            # Try SQLAlchemy first (better pandas support)
            if self.sqlalchemy_engine:
                try:
                    # SQLAlchemy uses different parameter style
                    sqlalchemy_query = query.replace('%s', ':limit')
                    # 🔧 FIXED: Use query variable instead of sqlalchemy_query
                    df = pd.read_sql_query(sqlalchemy_query, self.sqlalchemy_engine, params={'limit': limit})
                    
                    if len(df) > 0:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        print(f"✅ Retrieved {len(df)} recent transactions via SQLAlchemy")
                    else:
                        print("⚠️ No recent transactions found")
                    return df
                    
                except Exception as e:
                    print(f"⚠️ SQLAlchemy query failed: {e}")
            
            # Fallback to psycopg2
            df = pd.read_sql_query(query, self.connection, params=(limit,))

            if len(df) > 0:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"✅ Retrieved {len(df)} recent transactions via psycopg2")
            else:
                print("⚠️ No recent transactions found")

            return df

        except Exception as e:
            print(f"❌ Error retrieving recent transactions: {e}")
            return pd.DataFrame()

    def get_all_transactions_for_ml(self) -> pd.DataFrame:
        """ENHANCED: Get ALL transactions for ML with SQLAlchemy support"""
        try:
            query = """
                SELECT timestamp, price, volume, rsi, amount_in, amount_out,
                       price_impact, profitable, input_token, output_token
                FROM transactions
                WHERE price IS NOT NULL AND price > 0 AND rsi IS NOT NULL
                ORDER BY timestamp ASC;
            """
            
            # Try SQLAlchemy engine first (better pandas support)
            if self.sqlalchemy_engine:
                try:
                    df = pd.read_sql_query(query, self.sqlalchemy_engine)
                    
                    if len(df) > 0:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        print(f"✅ Retrieved {len(df)} ALL transactions via SQLAlchemy engine")
                    else:
                        print("⚠️ No transactions found for ML training")
                    return df
                except Exception as e:
                    print(f"⚠️ SQLAlchemy ML query failed: {e}, falling back to psycopg2")
            
            # Fallback to original psycopg2 method
            df = pd.read_sql_query(query, self.connection)
            
            if len(df) > 0:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                print(f"✅ Retrieved {len(df)} ALL transactions via psycopg2 fallback")
            else:
                print("⚠️ No transactions found for ML training")

            return df

        except Exception as e:
            print(f"❌ Error retrieving ALL ML training data: {e}")
            return pd.DataFrame()

    def save_ml_model_info(self, model_info: Dict):
        """Save ML model performance info"""
        try:
            cursor = self.connection.cursor()

            # Deactivate old models of the same type
            cursor.execute("""
                UPDATE ml_models
                SET is_active = FALSE
                WHERE model_type = %s;
            """, (model_info.get('model_type', 'unknown'),))

            # Insert new model info
            cursor.execute("""
                INSERT INTO ml_models
                (model_name, model_type, performance_metrics, training_samples,
                 accuracy, r2_score, mae, model_file_path, is_active, last_trained)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                model_info.get('model_name', 'Unknown'),
                model_info.get('model_type', 'unknown'),
                psycopg2.extras.Json(model_info.get('metrics', {})),
                model_info.get('training_samples', 0),
                model_info.get('accuracy', 0),
                model_info.get('r2_score', 0),
                model_info.get('mae', 0),
                model_info.get('model_file_path', ''),
                True,  # is_active
                datetime.utcnow()
            ))

            model_id = cursor.fetchone()[0]
            cursor.close()

            print(f"✅ ML model info saved (ID: {model_id})")
            return model_id

        except Exception as e:
            print(f"❌ Error saving ML model info: {e}")
            return None

    def get_ml_model_performance(self) -> Dict:
        """Get active ML models performance"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT model_name, model_type, accuracy, r2_score, mae,
                       training_samples, last_trained
                FROM ml_models
                WHERE is_active = TRUE
                ORDER BY last_trained DESC;
            """)

            results = cursor.fetchall()
            cursor.close()

            performance = {}
            for row in results:
                model_name = row[0]
                performance[model_name] = {
                    'model_type': row[1],
                    'accuracy': float(row[2]) if row[2] else 0,
                    'r2': float(row[3]) if row[3] else 0,
                    'mae': float(row[4]) if row[4] else 0,
                    'training_samples': row[5] if row[5] else 0,
                    'last_trained': row[6].strftime('%Y-%m-%d %H:%M:%S') if row[6] else 'Never'
                }

            return performance

        except Exception as e:
            print(f"❌ Error getting ML performance: {e}")
            return {}

    def migrate_from_csv(self, csv_file_path: str):
        """Migrate existing CSV data to PostgreSQL"""
        try:
            if not os.path.exists(csv_file_path):
                print(f"⚠️ CSV file not found: {csv_file_path}")
                return False

            # Read CSV
            df = pd.read_csv(csv_file_path)
            print(f"📊 Found {len(df)} records in CSV")

            if len(df) == 0:
                print("⚠️ CSV file is empty")
                return False

            # Check if we already have data
            existing_count = self.get_transaction_count()
            if existing_count > 0:
                print(
                    f"⚠️ Database already has {existing_count} transactions. Skipping migration.")
                return True

            # Migrate data
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

            print(
                f"✅ Successfully migrated {migrated}/{len(df)} records from CSV")
            return True

        except Exception as e:
            print(f"❌ CSV migration failed: {e}")
            return False

    def get_transaction_count(self) -> int:
        """Get total number of transactions"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM transactions;")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Exception as e:
            print(f"❌ Error getting transaction count: {e}")
            return 0

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            stats = {}
            cursor = self.connection.cursor()

            # Transaction stats
            cursor.execute("SELECT COUNT(*) FROM transactions;")
            stats['total_transactions'] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM transactions WHERE profitable = TRUE;")
            stats['profitable_transactions'] = cursor.fetchone()[0]

            cursor.execute(
                "SELECT AVG(amount_out - amount_in) FROM transactions;")
            avg_pnl = cursor.fetchone()[0]
            stats['avg_pnl'] = float(avg_pnl) if avg_pnl else 0

            # ML model stats
            cursor.execute(
                "SELECT COUNT(*) FROM ml_models WHERE is_active = TRUE;")
            stats['active_ml_models'] = cursor.fetchone()[0]

            cursor.close()
            return stats

        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {}

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("✅ Database connection closed")
        if self.sqlalchemy_engine:
            self.sqlalchemy_engine.dispose()


# Global database manager instance
_db_manager = None


def get_db_manager():
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager