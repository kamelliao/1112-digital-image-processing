# DIP Homework Assignment #2
# Name: 廖政華
# ID #: r11922133
# email: r11922133@ntu.edu.tw

# Problem 1
python3 main.py sobel -i hw2_sample_images/sample1.png -o result1.png
python3 main.py sobel -i hw2_sample_images/sample1.png -o result2.png -t 75
python3 main.py canny -i hw2_sample_images/sample1.png -o result3.png
python3 main.py log -i hw2_sample_images/sample1.png -o result4.png
python3 main.py edge_crisp -i hw2_sample_images/sample2.png -o result5.png
python3 main.py canny -i result5.png -o result6.png
python3 main.py hough -i result6.png -o result7.png

# Problem 2
python3 main.py borzoi -i hw2_sample_images/sample3.png -o result8.png
python3 main.py popdog -i hw2_sample_images/sample5.png -o result9.png