fname = '/Users/ruairi.moran@equipmentshare.com/Projects/clone_raocp-parallel/raocp-parallel/json/timeScaling.csv';
data = readtable(fname);
N = data.Var1;
nx = data.Var2;
t = data.Var3;

semilogy(N, t, "LineWidth", 1.5);
fontsize(14, "points");
xlabel("N");
ylabel("time (ms)");
