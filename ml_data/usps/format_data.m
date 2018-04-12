load USPS
X = reshape(fea, 9298, 16, 16);
y = gnd;
save('-v6','USPS.mat','X','y')
