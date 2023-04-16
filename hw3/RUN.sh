# DIP Homework Assignment #3
# Name: 廖政華
# ID #: r11922133
# email: r11922133@ntu.edu.tw

# Problem 1
python3 main.py boundary -i hw3_sample_images/sample1.png -o result1.png
python3 main.py holefill -i hw3_sample_images/sample1.png -o result2.png
python3 main.py count -i hw3_sample_images/sample1.png -o resultc.png
python3 main.py open -i hw3_sample_images/sample1.png -o result3.png
python3 main.py close -i hw3_sample_images/sample1.png -o result4.png

# Problem 2
python3 main.py texture -i hw3_sample_images/sample2.png -o result5.png --n_clusters 4 -r 17 -p mean 
python3 main.py texture -i hw3_sample_images/sample2.png -o result6.png --n_clusters 4 -r 17 -p mean --pos
python3 image_quilting.py