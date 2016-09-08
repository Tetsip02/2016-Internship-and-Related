reset
#read in A,B and C
plot 'A.out' every ::0::0 using (A11 = $1, A12 = $2), \
      'A.out' every ::1::1 using (A21 = $1, A22 = $2), \
      'B.out' using (b1 = $1, b2 = $2), \
      'C.out' using (c = $1)

set terminal png
set output 'QDA_plot.png'

set xlabel "Feature 1"
set ylabel "Feature 2"

plot (-(((A12+A21) * x) + b2) + sqrt(((A12+A21) + b2)**2 - 4 * A22 * (A11*(x**2) + b1*x + c)))/(2*A22) title 'boundary1', \
        (-((A12+A21)*x+b2) - sqrt(((A12+A21)+b2)**2-4*A22*(A11*x**2 + b1*x +c)))/(2*A22) title 'boundary2', \
        'Class0.out' title 'Species 1', \
        'Class1.out' title 'Species 2'
