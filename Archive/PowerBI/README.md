# Microsoft PowerBI
I am using Microsoft PowerBI to analyze my data. My sample file is in the directory called "[March_Madness.pbix](/March_Madness.pbix)". All files that end in "*.pbix" are a PowerBI file.

## Information
In each folder, there are some assorted stats for each team each season, which was collected using sportsipy. [March_Madness.pbix](/Archive/PowerBI/March_Madness.pbix) is my PowerBI File. [Power_Query](/Archive//PowerBI/Power_Query) is the folder with the large PowerQueries that are used to operate the PowerBI calculations and rankings


#### Installing PowerBI
An installer for Microsoft PowerBI can be found here: https://powerbi.microsoft.com/en-us/downloads/


#### Other Notes
For some reason or another, PowerBI does ***NOT*** support relative file paths. So unfortunately, you will have to go into the PowerQuery Editor, and change the file path. 

#### Steps to change File Path
1. Open the .pbix file
2. Click on Transform Data
3. Find the query that is called "FilePath"
4. Update this to the root directory that all of your code is stored in
5. Press "Close & Apply". All of your data should now be correct and up to date