
"""Programmer: Lindsey Bodenbender
Class: CPSC 322 Fall 2024
11/18/2024

Description: This program contains a class MyPyTable that contains functions to represent a 2D table of data with column names"""

import copy
import csv
from tabulate import tabulate
from mysklearn import myutils

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        if not isinstance(col_identifier, (str, int)):
            raise ValueError(f"Invalid column identifier: {col_identifier} must be string or integer type")
        column = []
        if include_missing_values:
            # get the index of col_identifier
            col_index = self.column_names.index(col_identifier)
            for row in range(len(self.data)):
                for col in range(len(self.data[row])):
                    if col == col_index:
                        column.append(self.data[row][col])
        else:
            self.remove_rows_with_missing_values()
            # get the index of col_identifier
            col_index = self.column_names.index(col_identifier)
            for row in range(len(self.data)):
                for col in range(len(self.data[row])):
                    if col == col_index:
                        column.append(self.data[row][col])
        return column

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try:
                    numeric_val = float(self.data[i][j])  
                    self.data[i][j] = numeric_val
                except ValueError as e:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        # create a new list without the specified rows
        self.data = [row for i, row in enumerate(self.data) if i not in row_indexes_to_drop]
        # return [row for i, row in enumerate(self.data) if i not in row_indexes_to_drop]
        return self

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        table = []
        # open the file
        with open(filename, 'r', encoding='utf-8') as infile:
            # process the file
            reader = csv.reader(infile)
            header = next(reader)
            for row in reader:
                table.append(row)
            # close the file
            infile.close()
        # self.__init__(header, table)
        self.column_names = header
        self.data = table
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        key_indexes = []
        seen = []
        temp_row = []

        # get indexes of key_column_names
        for i in self.column_names:
            if i in key_column_names:
                key_indexes.append(self.column_names.index(i))

        # create list of seen data
        for row in self.data:
            temp_row = []
            for i in row:
                col_index = row.index(i)
                if col_index in key_indexes:
                    temp_row.append(i)
            seen.append(temp_row)

        duplicate_indexes = []
        
        # create a list of the items that are duplicates
        instance_count = 0
        for i, val in enumerate(seen):
            for j in range(len(seen)):
                if val == seen[j] and j > i and j not in duplicate_indexes:
                    instance_count += 1
                    duplicate_indexes.append(j)
                    
        return duplicate_indexes
    
    def extract_key_from_row(self, row, header, key_column_names):
        key_indexes = []
        # get indexes of key_column_names
        for i in header:
            if i in key_column_names:
                key_indexes.append(header.index(i))

        # create list of key data
        key_data = []
        for i in row:
            col_index = row.index(i)
            if col_index in key_indexes:
                key_data.append(i)
        return key_data

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        row_indexes_to_drop = []
        row_index = 0
        for row in self.data:
            row_index += 1
            for i in row:
                if not i or i == 'NA': # if item in row is missing
                    row_indexes_to_drop.append(row_index - 1)
        self.drop_rows(row_indexes_to_drop)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        for row in range(len(self.data)):
            for i in range(len(self.data[row])):
                if not self.data[row][i] or self.data[row][i] == "NA" and i == self.column_names.index(col_name):
                    self.data[row][i] = self.calculate_average(col_name)

    def calculate_average(self, col_name):
        column = self.get_column(col_name)
        col_sum = 0
        count = 0
        for i in column:
            if isinstance(i, float):
                col_sum += i
                count += 1
            average = col_sum / count
        return average 

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        summary_stats = []

        for i in col_names:
            row = []
            column = self.get_column(i, False) # don't include missing values
            if column == []:
                print("Column empty can't compute stats")
                summary_table = MyPyTable(column_names=["attribute", "min", "max", "mid", "avg", "median"], data=summary_stats)
                return summary_table
            if isinstance(column[0], int) or isinstance(column[0], float):
                col_min = self.find_min(column)
                col_max = self.find_max(column)
                mid = self.find_mid(column)
                avg = self.calculate_average(i)
                median = self.find_median(column)

                row.append(i)
                row.append(col_min)
                row.append(col_max)
                row.append(mid)
                row.append(avg)
                row.append(median)

                summary_stats.append(row)
            else:
                summary_stats.append(row)

        summary_table = MyPyTable(column_names=["attribute", "min", "max", "mid", "avg", "median"], data=summary_stats)
        return summary_table
    
    def find_min(self, column):
        """ Function to find the minimum value of the column"""
        col_min = column[0]
        for i in column:
            if isinstance(i, int) or isinstance(i, float):
                if i < col_min:
                    col_min = i
        return col_min
    
    def find_max(self, column):
        """ Function to find the maximum value of the column"""
        col_max = column[0]
        for i in column:
            if isinstance(i, int) or isinstance(i, float):
                if i > col_max:
                    col_max = i
        return col_max
    
    def find_mid(self, column):
        """ Function to find the mid value of the column"""

        mid = (self.find_max(column) + self.find_min(column)) / 2
        return mid
    
    def find_median(self, column):
        """ Function to find the median value of the column"""
        if isinstance(column[0], int) or isinstance(column[0], float):
            col_sorted = sorted(column)
            mid_index = len(col_sorted) // 2
        # if the length of the sorted column is divisible by 2 (if its even)
            # the median is equal to the average of the values at the two middle indexes
            if len(col_sorted) % 2 == 0:
                median = self.calc_avg_two_nums(col_sorted[mid_index], col_sorted[mid_index - 1])
            else:
                median = col_sorted[mid_index]
        return median

    def calc_avg_two_nums(self, num1, num2):
        """ Function to calculate the average of two numbers"""
        return (num1 + num2) / 2

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # create a list to store resulting joined data
        joined_data = []

        # create a combined header with columns from both tables
        combined_header = self.column_names[:]
        for col in other_table.column_names:
            if col not in key_column_names:
                combined_header.append(col)

        # extract keys from both datasets, compare to one another
        for self_row in self.data:
            self_key = self.extract_key_from_row(self_row, self.column_names, key_column_names)

            for other_row in other_table.data:
                other_key = self.extract_key_from_row(other_row, other_table.column_names, key_column_names)
                # if key items match
                if all(i in self_key for i in other_key):
                    # if match combine rows
                    combined_row = self_row[:] # start with self_row
                    for i, value in enumerate(other_row):
                        # add other column values to combined row if not in the key column (already added these elements when starting w self_row)
                        if other_table.column_names[i] not in key_column_names:
                            combined_row.append(value)
                    joined_data.append(combined_row)

        return MyPyTable(combined_header, joined_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        joined_data = []

        # combine header
        # starting with all column names in self
        combined_header = self.column_names[:]
        # append other table's column names (excluding key column names)
        for col in other_table.column_names:
            if col not in key_column_names:
                combined_header.append(col)

        # keep track of the matched rows
        matched_rows = []

        for self_row in self.data:
            self_key = self.extract_key_from_row(self_row, self.column_names, key_column_names)
            match = False
            for other_row in other_table.data:
                other_key = other_table.extract_key_from_row(other_row, other_table.column_names, key_column_names)
                # if match
                if all(i in self_key for i in other_key):
                    match = True
                    combined_row = self_row[:] # start with self row
                    for i, value in enumerate(other_row):
                        if other_table.column_names[i] not in key_column_names:
                            combined_row.append(value)
                    joined_data.append(combined_row)
                    matched_rows.append(tuple(other_row))
            if not match:
                # start with self row and append NA's
                combined_row = self_row[:] + ["NA"] * (len(other_table.column_names) - len(key_column_names))
                joined_data.append(combined_row)
        # handle rows in other_table that were not matched
        for row in other_table.data:
            if tuple(row) not in matched_rows:
                # initialize combined row w all 0's
                combined_row = [0] * len(combined_header)

                for i in combined_header:
                    if i in other_table.column_names:
                        # combined_row.append(row[combined_header.index(i)])
                        combined_row[combined_header.index(i)] = row[other_table.column_names.index(i)]
                    elif i in self.column_names:
                        combined_row[combined_header.index(i)] = "NA"
                joined_data.append(combined_row)

        return MyPyTable(combined_header, joined_data)
