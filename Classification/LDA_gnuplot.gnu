reset
set terminal png
set output 'LDA_plot.png'
#load first row first column of p.out into p
p = system("head -n1 p.out | awk '{print $1}'")
q = system("head -n1 q.out | awk '{print $1}'")
y(x) = p * x + q

set xlabel "Feature 1"
set ylabel "Feature 2"

plot y(x) title "decision boundary", \
    'Class0.out' title 'Species 1', \
    'Class1.out' title "Species 2"
