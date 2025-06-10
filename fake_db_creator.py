import sqlite3
import random
from datetime import datetime, timedelta
from faker import Faker

# Setup
fake = Faker('en_IN')
db_path = "grocery_simulation.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Drop existing tables if they exist
cursor.executescript("""
DROP TABLE IF EXISTS Users;
DROP TABLE IF EXISTS Orders;
DROP TABLE IF EXISTS Products;
DROP TABLE IF EXISTS Wallet;
""")

# Create new tables
cursor.executescript("""
CREATE TABLE Users (
    UserID INTEGER PRIMARY KEY,
    UserName TEXT,
    UserPhone TEXT,
    UserAddress TEXT,
    WalletID INTEGER
);

CREATE TABLE Products (
    ProductID INTEGER PRIMARY KEY,
    ProductName TEXT,
    ProductPrice REAL,
    ProductQuantity INTEGER
);

CREATE TABLE Orders (
    OrderID INTEGER PRIMARY KEY,
    UserID INTEGER,
    OrderItems TEXT,
    OrderTotal REAL,
    OrderStatus TEXT,
    OrderPlacedTime TEXT,
    OrderDeliveredTime TEXT,
    FOREIGN KEY(UserID) REFERENCES Users(UserID)
);

CREATE TABLE Wallet (
    WalletID INTEGER,
    Amount REAL,
    Datetime TEXT
);
""")

# Add 10 Users
users = []
for i in range(10):
    name = fake.name()
    phone = fake.phone_number()
    address = fake.address().replace("\n", ", ")
    wallet_id = 1000 + i
    users.append((i + 1, name, phone, address, wallet_id))
cursor.executemany("INSERT INTO Users VALUES (?, ?, ?, ?, ?)", users)

# Add 50 Products
product_names = [
    "Basmati Rice", "Toor Dal", "Wheat Flour", "Sugar", "Salt", "Groundnut Oil", "Mustard Oil",
    "Coriander Powder", "Turmeric", "Chilli Powder", "Black Pepper", "Cumin Seeds", "Garam Masala",
    "Green Tea", "Coffee", "Tomato Ketchup", "Pickle", "Atta Biscuits", "Marie Gold", "Milk",
    "Curd", "Paneer", "Butter", "Ghee", "Honey", "Bread", "Jam", "Maggi", "Poha", "Oats",
    "Rava", "Moong Dal", "Chana Dal", "Besan", "Jaggery", "Dry Fruits", "Cashews", "Almonds",
    "Raisins", "Green Gram", "Brown Rice", "Sooji", "Dalia", "Papad", "Sabudana", "Tea",
    "Cornflakes", "Chips", "Namkeen", "Ice Cream"
]

products = []
for i, name in enumerate(random.sample(product_names, 50), start=1):
    price = round(random.uniform(10, 200), 2)
    quantity = random.randint(10, 100)
    products.append((i, name, price, quantity))
cursor.executemany("INSERT INTO Products VALUES (?, ?, ?, ?)", products)

# Add Orders and Wallet Transactions for 7 Days
order_id = 1
wallet_entries = []
orders = []

for day in range(7):
    base_time = datetime.now() - timedelta(days=(6 - day))
    for user in users:
        if random.random() < 0.8:  # 80% chance they place order that day
            user_id = user[0]
            wallet_id = user[4]
            num_items = random.randint(1, 5)
            selected_products = random.sample(products, num_items)
            product_ids = [str(p[0]) for p in selected_products]
            total = round(sum(p[2] for p in selected_products), 2)
            placed_time = base_time + timedelta(hours=random.randint(8, 20))
            delivered_time = base_time + timedelta(days=1, hours=random.randint(5, 8))
            status = "Delivered"

            orders.append((order_id, user_id, ','.join(product_ids), total, status, placed_time.isoformat(), delivered_time.isoformat()))
            wallet_entries.append((wallet_id, -total, placed_time.isoformat()))

            order_id += 1

cursor.executemany("INSERT INTO Orders VALUES (?, ?, ?, ?, ?, ?, ?)", orders)
cursor.executemany("INSERT INTO Wallet VALUES (?, ?, ?)", wallet_entries)

conn.commit()
conn.close()

print(f"Database created: {db_path}")
print("Fake grocery database created successfully with 10 users, 50 products, and orders for 7 days.")
