#!/usr/bin/env python3
"""
Script to convert decimal commas to points and semicolon delimiters to commas in a CSV file.
"""

# Define the path to the CSV file here
PATH = 'data/Info_Sheets/All_Data_Renamed_overview.csv'


def main():
    # Read the entire file as text
    with open(PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Replace all decimal commas with periods
    content = content.replace(',', '.')
    # 2. Replace all semicolons (original delimiters) with commas
    content = content.replace(';', ',')

    # Write the transformed content back to the same file
    with open(PATH, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Converted separators in '{PATH}' (','→'.' and ';'→',').")


if __name__ == '__main__':
    main()

