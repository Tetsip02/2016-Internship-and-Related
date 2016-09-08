reset
set terminal png
set output 'QDA_plot.png'

set multiplot
#dataplot
set xlabel "Sepal length"
set ylabel "Sepal width"
set size 1,1
plot 'Class0.out' title "Setosa", \
    'Class1.out' title "versicolor"
#superpose contour plot
unset xlabel
unset ylabel
unset key
unset border
unset xtics
unset ytics
set dgrid3d
set parametric
set contour base
set view 0,0,1
unset surface
set cntrparam levels discrete 0
splot 'QDA_dat.out' using 1:2:3 with lines
unset multiplot
