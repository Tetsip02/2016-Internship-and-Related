reset
set terminal png
set output 'QDA_plot.png'

set xlabel "Feature 1"
set ylabel "Feature 2"

plot 'QDA_dat.out' with lines title "decision boundary", \
    'Class0.out' title "Species 1", \
    'Class1.out' title "Species 2"
