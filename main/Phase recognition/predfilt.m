% filter narx predictions using heaviside approach

function yp1f = predfilt(yp1,predLength)

yp1f = zeros(1,predLength);
for i = 1: predLength
    if 1.5 > yp1{i} && yp1 {i} > 0
       yp1f(i) = 1;
    elseif 2.5 > yp1{i} && yp1 {i} > 1.5
       yp1f(i) = 2;
    elseif 3.5 > yp1{i} && yp1 {i} > 2.5
       yp1f(i) = 3;
    elseif 4.5 > yp1{i} && yp1 {i} > 3.5
       yp1f(i) = 4;
    elseif 5.5 > yp1{i} && yp1 {i} > 4.5
       yp1f(i) = 5;
    elseif 6.5 > yp1{i} && yp1 {i} > 5.5
       yp1f(i) = 6;
    elseif 10 > yp1{i} && yp1 {i} > 6.5
       yp1f(i) = 7;
    end
end
end 