import mysql.connector
from mysql.connector import Error

# Database connection parameters
hostname = 'shrimpdata.mysql.database.azure.com'
port = 3306
username = 'Bariflolabs'
password = 'Bfl@2024'
database_name = 'shrimp_data'  # Replace with the name of the database you want to create

try:
    # Establish the connection to the server (without specifying a database)
    connection = mysql.connector.connect(
        host=hostname,
        user=username,
        password=password,
        port=port
        #ssl_ca=ssl_ca,  # Uncomment if you are using SSL
        # ssl_verify_cert=True  # Uncomment if you want SSL verification
    )

    if connection.is_connected():
        print("Connected to the MySQL server")

        # Create a cursor object
        cursor = connection.cursor()

        # Create the new database
        cursor.execute(f"CREATE DATABASE {database_name}")
        print(f"Database '{database_name}' created successfully")

except Error as e:
    print(f"Error: {e}")
finally:
    # Close the cursor and connection
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed")
