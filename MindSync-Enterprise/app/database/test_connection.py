from app.database.connection import test_connection

if __name__ == "__main__":
    try:
        test_connection()
        print("MongoDB Atlas connected successfully!")
    except Exception as error:
        print("MongoDB connection failed.")
        print("Error:", error)