# cs123proj
Alyssa Blackburn, Lyle Kim and Jessica Lu

Files: 

Main directory:
elasticgridsearch.py: Our main function and contains the District class and map reduce command. Runs on an EC2 instance or with EMR and saves to s3 buckets using boto3. Given a state's CSV file with all its blocks, redistricts according to population and graphs to a png file, which it then saves to an s3 bucket.
 
parse_data.py: Takes a state's 00001 uf1 and geo uf1 file and parses for the relevant data: all the census blocks in the state, with a unique ID, latitude, longitude and population. Simply extract these two files from the data directory and run with the state's abbreviation (e.g. "il" for Illinois) and your parsed csv will be saved in the main directory. Our finished parsed csvs are located in the directory "statescsv." (The data directory has since been deleted as uploading it would just basically be a big data dump).

state.csv:
to be fed into the mapper: each line has the name of the CSV where the block information is held, along with the number of distircts for that state.

Data directory (now deleted):
All the 00001 uf1 and geo uf1 zip files from the AWS public dataset. 

actual_districts directory: 
Contains pictures of the of what the actual 2000 census districts looked like from the shapefiles. 

districtpics directory: 
pictures of our redistricted (or partially redistricted) states. 

statecsv directory:
All the state CSV files after it has been parsed by parse_data.py as described above. 

Alternative methods directory: 

linear_search.py: This file implements the most basic method of our algorithm. The alogorithm runs as follows:
1) remove centroids from the input data
2) identifies the district with smallest population
3) calculates the distance between the centroid of smallest population
   and every block in data (which is numpy array of blocks)
4) identify the nearest block from the centroid 
5) the district absorbs the nearest block, recalculates the population
6) repeat 2~5 until every block has been assigned

linear_search_multiprocessing.py: This file implements python multiprocessing library to linear_search.py.
Initially, we used a function called pool.map that automatically divides up the data and processors behind the scene. But using this method, it interestingly slowed down the entire process. So instead, after consulting with Professor Wachs, we divided up the data
into N chunks manually, and did the same with processors as well, and feeded sub_data into each processors. This resulted in expediting linear_search by approximately N times faster.

linear_search_pool.py:
This file is almost identical to linear_search_multiprocessing.py, but uses pool.map method instead of manually splitting up the data. It turned out that this method slows down the algorithm rathr than expediting.

grid_search_multiprocessing.py:
This file is almost identical to grid_search.py, but it implements multiprocessing method that we tested in linear_search_multiprocessing.py.
However, interestingly, this method turned out to be slower than regular gird_search.py even though multiprocessing method did expedite linear_search.py significantly.
We speculate that the reason is since grid_search.py is already significnatly optimized and only loops through a small subset of the entire data (numpy array of blocks), there might not be enough blocks to merit from launching several processors and dividing up the data into chunks. (i.e. dividing up the data and launching several processors might be more expensive)




