# DIP Homework Assignment #1
# Name: 廖政華
# ID #: r11922133
# email: r11922133@ntu.edu.tw

# Problem 0
python3 main.py grayscale -i hw1_sample_images/sample1.png -o result1.png
python3 main.py vflip -i result1.png -o result2.png

# Problem 1
python3 main.py dec_brightness -g -i hw1_sample_images/sample2.png -o result3.png
python3 main.py inc_brightness -g -i result3.png -o result4.png
python3 plot.py -i hw1_sample_images/sample2.png result3.png result4.png -o hist-2c.png

python3 main.py global_hist_equalize -g -i hw1_sample_images/sample2.png -o result5.png
python3 main.py global_hist_equalize -g -i result3.png -o result6.png
python3 main.py global_hist_equalize -g -i result4.png -o result7.png
python3 plot.py -i result5.png result6.png result7.png -o hist-2d.png

python3 main.py local_hist_equalize -g -i hw1_sample_images/sample2.png -o result8.png
python3 main.py local_hist_equalize -g -i result3.png -o result9.png
python3 main.py local_hist_equalize -g -i result4.png -o result10.png
python3 plot.py -i result8.png result9.png result10.png -o hist-2e.png

python3 main.py transfer_func -g -i hw1_sample_images/sample2.png -o result11.png
python3 plot.py -i result11.png -o hist-2f.png

# Problem 2
python3 main.py spatial_filter -g -i hw1_sample_images/sample4.png -o result12.png
python3 main.py pmed_filter -g -i hw1_sample_images/sample5.png -o result13.png
python3 main.py psnr -g -i hw1_sample_images/sample3.png -o result12.png
python3 main.py psnr -g -i hw1_sample_images/sample3.png -o result13.png