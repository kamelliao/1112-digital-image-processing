# DIP Homework Assignment #4
# Name: 廖政華
# ID #: r11922133
# email: r11922133@ntu.edu.tw

# Problem 1
python3 main.py dither -i hw4_sample_images/sample1.png -o result1.png
python3 main.py dither -i hw4_sample_images/sample1.png -o result2.png -r 256
python3 main.py error_diffusion -i hw4_sample_images/sample1.png -o result3.png -m floyd-steinberg
python3 main.py error_diffusion -i hw4_sample_images/sample1.png -o result4.png -m jarvis

# Problem 2
python3 main.py image_sample -i hw4_sample_images/sample2.png -o result5.png
python3 main.py gaussian_highpass -i hw4_sample_images/sample2.png -o result6.png
python3 main.py remove_pattern -i hw4_sample_images/sample3.png -o result7.png