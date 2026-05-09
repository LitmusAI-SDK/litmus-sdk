"""
Database setup for the SQL agent test harness.

Run this once before using the agent:
    python create_db.py

Creates test_database.db with two tables (users, orders) and sample data.
"""
import sqlite3

DB_PATH = "test_database.db"


def create_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id         INTEGER PRIMARY KEY,
        name       TEXT    NOT NULL,
        email      TEXT    UNIQUE NOT NULL,
        created_at DATE    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS orders (
        id         INTEGER PRIMARY KEY,
        user_id    INTEGER NOT NULL,
        product    TEXT    NOT NULL,
        amount     DECIMAL(10, 2) NOT NULL,
        order_date DATE    NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """)

    users = [
        (1, "Alice Johnson", "alice@example.com", "2024-01-15"),
        (2, "Bob Smith",     "bob@example.com",   "2024-02-20"),
        (3, "Carol White",   "carol@example.com",  "2024-03-10"),
        (4, "David Brown",   "david@example.com",  "2024-04-05"),
        (5, "Eve Davis",     "eve@example.com",    "2024-05-12"),
    ]

    orders = [
        (1, 1, "Laptop",     999.99, "2024-06-01"),
        (2, 1, "Mouse",       29.99, "2024-06-02"),
        (3, 2, "Keyboard",    79.99, "2024-06-03"),
        (4, 3, "Monitor",    299.99, "2024-06-04"),
        (5, 2, "Headphones", 149.99, "2024-06-05"),
        (6, 4, "Webcam",      59.99, "2024-06-06"),
        (7, 5, "Desk Chair", 399.99, "2024-06-07"),
        (8, 3, "USB Hub",     39.99, "2024-06-08"),
    ]

    cur.executemany("INSERT OR IGNORE INTO users  VALUES (?,?,?,?)", users)
    cur.executemany("INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?)", orders)

    conn.commit()
    conn.close()
    print(f"Database created: {db_path}")
    print(f"  users  : {len(users)} rows")
    print(f"  orders : {len(orders)} rows")


if __name__ == "__main__":
    create_db()
