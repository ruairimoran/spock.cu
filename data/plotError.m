clf;
clear;
fname = './errorCache.json';
fid = fopen(fname);
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
json = jsondecode(str);
maxIters = json.maxIters;
tol = json.tol;
err0 = json.err0;
err1 = json.err1;
err2 = json.err2;
semilogy(err0);
hold on;
semilogy(err1);
semilogy(err2);
legend('\xi_{0}', '\xi_{1}', '\xi_{2}')
title('Vanilla CP errors')
xlabel('iteration')
ylabel('magnitude')
