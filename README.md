Inventory Management System
This project is an Inventory Management System implemented in Python, designed to manage food items with expiry dates. The system provides functionalities for adding, editing, deleting items, searching by barcode or name, viewing near expiry items, and saving inventory data to a CSV file.

Project Structure
The project consists of the following files and directories:

main.py: Contains the main program logic for interacting with the inventory.
invents1.py: Defines the Inventory class responsible for managing the inventory of food items.
items.py: Defines the FoodItem class representing individual food items with attributes such as name, category, quantity, barcode, and expiry date.
Features
Add Item: Allows adding a new food item to the inventory with details including name, category, quantity, barcode, and expiry date.

Edit Item: Enables editing the quantity of an existing item in the inventory using its barcode.

Delete Item: Deletes an item from the inventory based on its barcode.

Search Item: Provides functionality to search for an item by either its barcode or name.

View Near Expiry Items: Displays items in the inventory that are nearing their expiry date, based on a specified threshold of days.

Save Inventory to CSV: Saves the current inventory data to a CSV file for future reference or backup.

Challenges Faced
Initially, this project encountered challenges related to module imports:

Module Organization: Structuring the project with separate files (main.py, invents1.py, items.py) initially caused confusion with module imports.
Import Errors: Troubleshooting errors such as "No module named 'invents1'" and "ModuleNotFoundError" due to incorrect paths and module naming.
