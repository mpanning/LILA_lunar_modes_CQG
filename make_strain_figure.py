"""
Plot up strain of likely lunar seismicity, highlight mode frequencies, and
compare with noise of strain and seismometer systems
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, next_fast_len
from matplotlib import mlab
import math
from obspy import Stream, Trace, UTCDateTime

fileout = "strain_figure.png"
modes_root = "./modes_300mHz"
mdl = "weber2010hr"
DMQ_path = "800km_3hrs"
SMQ_path = "30km_3hrs"

DMQ_moment = 1.e13
DMQ_max_moment = math.pow(10, 13.8)
DMQ_max_scale = DMQ_max_moment/DMQ_moment
SMQ_moment = 3.98e16
SMQ_scale = 1.e-2 # To account for me mistakenly equating mb to Mw

# LILA strainmeter sensitivities
# LILAlow = "nomodes-conservative.dat"
# LILAmedium = "nomodes-baseline.dat"
# LILAhigh = "nomodes-ambitious.dat"
LILA_Pioneer = "lilanoise-10km-0.5W-nomodes.dat"
LILA_Horizon = "lilanoise-69km-0.5W-nomodes.dat"
lilafactors = [2.0, np.sqrt(3.0)]

distances = [5.0, 5.016, 5.033, 20.0, 20.016, 20.033, 40.0, 40.016, 40.033,
             90.0, 90.016, 90.033]
# These are zero-indexed to match the distance array, but the station names
# output by nms are 1-indexed
stn1 = 6
stn2 = 8

fmax = 0.3
rad = 1737100 # Radius of Moon in meters
iftrim = True
trimlength = 10800

# MSEEDS directory
mseeds_path = "./mseeds"

#PSD parameters
nfft = 8192
noverlap = 4096

# Obspy stuff
network = 'XA'
reftime = UTCDateTime(2025,1,1)

# Plot variables
hmin = 1.e-22
hmax = 1.e-9

# Calculate strain for DMQ (representative and largest)
stdmq1 = Stream()
stdmq2 = Stream()
stdmq_strain = Stream()

for comp in ['Z', 'N', 'E']:
    filename = "{}/{}/U{}_{:04d}".format(modes_root, DMQ_path, comp, stn1+1)
    df1 = pd.read_fwf(filename, header=None, names=['Time', comp])
    filename = "{}/{}/U{}_{:04d}".format(modes_root, DMQ_path, comp, stn2+1)
    df2 = pd.read_fwf(filename, header=None, names=['Time', comp])
    delta = df1['Time'].values[1]-df1['Time'].values[0]
    tr1 = Trace(data=df1[comp].values)
    tr1.stats['delta'] = delta
    tr1.stats['channel'] = "MH{}".format(comp)
    stnname = "DMQ{:02d}".format(stn1+1)
    tr1.stats['network'] = network
    tr1.stats['station'] = stnname
    tr1.stats['starttime'] = reftime + df1['Time'].values[0]
    stdmq1 = stdmq1 + tr1
    tr2 = Trace(data=df2[comp].values)
    tr2.stats['delta'] = delta
    tr2.stats['channel'] = "MH{}".format(comp)
    stnname = "DMQ{:02d}".format(stn2+1)
    tr2.stats['network'] = network
    tr2.stats['station'] = stnname
    tr2.stats['starttime'] = reftime + df2['Time'].values[0]
    stdmq2 = stdmq2 + tr2

#Low pass filter to limit some ringing
fcorner = 0.95*fmax
stdmq1.filter('lowpass', freq=fcorner)
stdmq2.filter('lowpass', freq=fcorner)
if iftrim:
    stime = reftime
    etime = stime+trimlength
    stdmq1.trim(starttime=stime, endtime=etime)
    stdmq2.trim(starttime=stime, endtime=etime)

# stdmq1.plot()
# stdmq2.plot()

# Write them to mseed files
filename = "{}/fmax{:04d}_{}_DMQ.mseed".format(mseeds_path, int(fmax*1000),
                                               stdmq1[0].stats['station'])
stdmq1.write(filename)
filename = "{}/fmax{:04d}_{}_DMQ.mseed".format(mseeds_path, int(fmax*1000),
                                               stdmq2[0].stats['station'])
stdmq2.write(filename)

# Calculate linear strain as the difference in E component
# (EE component of strain)
ddist = (distances[stn2] - distances[stn1])*rad*math.pi/180.
stdata = (stdmq2[2].data - stdmq1[2].data)/ddist
tr = Trace(data=stdata)
tr.stats['delta'] = delta
tr.stats['channel'] = 'MSE'
tr.stats['starttime'] = stdmq1[2].stats['starttime']
stnname = "D{:02d}{:02d}".format(stn1+1, stn2+1)
tr.stats['station'] = stnname
stdmq_strain = stdmq_strain + tr

filename = "{}/dmqstrain_{:02d}_{:02d}_DMQ.mseed".format(mseeds_path, stn1+1,
                                                        stn2+1)
stdmq_strain.write(filename)

# Scale typical DMQ up to the max observed DMQ moment
stdmqmax1 = stdmq1.copy()
for tr in stdmqmax1:
    tr.data = DMQ_max_scale*tr.data
stdmqmax2 = stdmq2.copy()
for tr in stdmqmax2:
    tr.data = DMQ_max_scale*tr.data

# stdmqmax1.plot()
# stdmqmax2.plot()

stdmqmax_strain = stdmq_strain.copy()
for tr in stdmqmax_strain:
    tr.data = DMQ_max_scale*tr.data

# Calculate strain for SMQ
stsmq1 = Stream()
stsmq2 = Stream()
stsmq_strain = Stream()

for comp in ['Z', 'N', 'E']:
    filename = "{}/{}/U{}_{:04d}".format(modes_root, SMQ_path, comp, stn1+1)
    df1 = pd.read_fwf(filename, header=None, names=['Time', comp])
    filename = "{}/{}/U{}_{:04d}".format(modes_root, SMQ_path, comp, stn2+1)
    df2 = pd.read_fwf(filename, header=None, names=['Time', comp])
    delta = df1['Time'].values[1]-df1['Time'].values[0]
    tr1 = Trace(data=df1[comp].values*SMQ_scale)
    tr1.stats['delta'] = delta
    tr1.stats['channel'] = "MH{}".format(comp)
    stnname = "SMQ{:02d}".format(stn1+1)
    tr1.stats['network'] = network
    tr1.stats['station'] = stnname
    tr1.stats['starttime'] = reftime + df1['Time'].values[0]
    stsmq1 = stsmq1 + tr1
    tr2 = Trace(data=df2[comp].values*SMQ_scale)
    tr2.stats['delta'] = delta
    tr2.stats['channel'] = "MH{}".format(comp)
    stnname = "SMQ{:02d}".format(stn2+1)
    tr2.stats['network'] = network
    tr2.stats['station'] = stnname
    tr2.stats['starttime'] = reftime + df2['Time'].values[0]
    stsmq2 = stsmq2 + tr2

#Low pass filter to limit some ringing
fcorner = 0.95*fmax
stsmq1.filter('lowpass', freq=fcorner)
stsmq2.filter('lowpass', freq=fcorner)
if iftrim:
    stime = reftime
    etime = stime+trimlength
    stsmq1.trim(starttime=stime, endtime=etime)
    stsmq2.trim(starttime=stime, endtime=etime)

# stsmq1.plot()
# stsmq2.plot()

# Write them to mseed files
filename = "{}/fmax{:04d}_{}_SMQ.mseed".format(mseeds_path, int(fmax*1000),
                                               stdmq1[0].stats['station'])
stsmq1.write(filename)
filename = "{}/fmax{:04d}_{}_SMQ.mseed".format(mseeds_path, int(fmax*1000),
                                               stdmq2[0].stats['station'])
stsmq2.write(filename)

# Calculate linear strain as the difference in E component
# (EE component of strain)
ddist = (distances[stn2] - distances[stn1])*rad*math.pi/180.
stdata = (stsmq2[2].data - stsmq1[2].data)/ddist
tr = Trace(data=stdata)
tr.stats['delta'] = delta
tr.stats['channel'] = 'MSE'
tr.stats['starttime'] = stsmq1[2].stats['starttime']
stnname = "S{:02d}{:02d}".format(stn1+1, stn2+1)
tr.stats['station'] = stnname
stsmq_strain = stsmq_strain + tr

filename = "{}/smqstrain_{:02d}_{:02d}_SMQ.mseed".format(mseeds_path, stn1+1,
                                                        stn2+1)
stsmq_strain.write(filename)

# Read in mode data and get mode frequencies below 10 mHz
modefile = "{}/{}S.csv".format(modes_root, mdl)
modedf = pd.read_csv(modefile, index_col=False)
maxf = 3
modedf_low = modedf.loc[(modedf['f'] < maxf) & (modedf['l'] > 0)]
modefreqs = modedf_low['f'].values * 1.e-3 # convert to Hz


# Plot up mlab and fft estimates (eventually only one, but this is just to
# verify calculation)
fs = stdmq_strain[0].stats['sampling_rate']
# (stdmqPxx, stdmqfreqs) = mlab.psd(stdmq_strain[0].data, Fs=fs, NFFT=nfft,
#                                   noverlap=noverlap, detrend='linear')
# char_strain_scale = np.sqrt(stdmqfreqs)
# plt.loglog(stdmqfreqs, np.multiply(char_strain_scale, np.sqrt(stdmqPxx)),
#            label="DMQ mlab")
stdmq_strain[0].detrend('demean')
npts = stdmq_strain[0].stats['npts']
nfft_scipy = 128*next_fast_len(npts)
scale = 1./np.sqrt(npts)
pos_freq = (nfft_scipy + 1 ) // 2
spec = fft(stdmq_strain[0].data, nfft_scipy)[:pos_freq]
freq = fftfreq(nfft_scipy, 1 / fs)[:pos_freq]
char_strain_scale = np.sqrt(freq)
plt.loglog(freq, scale*np.multiply(np.abs(spec), char_strain_scale),
           label="DMQ (typical)")

fs = stdmqmax_strain[0].stats['sampling_rate']
# (stdmqPxx, stdmqfreqs) = mlab.psd(stdmq_strain[0].data, Fs=fs, NFFT=nfft,
#                                   noverlap=noverlap, detrend='linear')
# char_strain_scale = np.sqrt(stdmqfreqs)
# plt.loglog(stdmqfreqs, np.multiply(char_strain_scale, np.sqrt(stdmqPxx)),
#            label="DMQ mlab")
stdmqmax_strain[0].detrend('demean')
npts = stdmqmax_strain[0].stats['npts']
nfft_scipy = 128*next_fast_len(npts)
scale = 1./np.sqrt(npts)
pos_freq = (nfft_scipy + 1 ) // 2
spec = fft(stdmqmax_strain[0].data, nfft_scipy)[:pos_freq]
freq = fftfreq(nfft_scipy, 1 / fs)[:pos_freq]
char_strain_scale = np.sqrt(freq)
plt.loglog(freq, scale*np.multiply(np.abs(spec), char_strain_scale),
           label="DMQ (largest)")

fs = stsmq_strain[0].stats['sampling_rate']
# (stdmqPxx, stdmqfreqs) = mlab.psd(stdmq_strain[0].data, Fs=fs, NFFT=nfft,
#                                   noverlap=noverlap, detrend='linear')
# char_strain_scale = np.sqrt(stdmqfreqs)
# plt.loglog(stdmqfreqs, np.multiply(char_strain_scale, np.sqrt(stdmqPxx)),
#            label="DMQ mlab")
stsmq_strain[0].detrend('demean')
npts = stsmq_strain[0].stats['npts']
nfft_scipy = 128*next_fast_len(npts)
scale = 1./np.sqrt(npts)
pos_freq = (nfft_scipy + 1 ) // 2
spec = fft(stsmq_strain[0].data, nfft_scipy)[:pos_freq]
freq = fftfreq(nfft_scipy, 1 / fs)[:pos_freq]
char_strain_scale = np.sqrt(freq)
plt.loglog(freq, scale*np.multiply(np.abs(spec), char_strain_scale),
           label="SMQ (largest)")

# add in lines for mode frequencies
plt.axvline(x=modefreqs[0], color='r', linestyle='--', linewidth=0.5,
            label="Modes (<{} mHz)".format(maxf))
for f in modefreqs[1:]:
    plt.axvline(x=f, color='r', linestyle='--', linewidth=0.5)

plt.xlim((1.e-5, 0.5))

# Add in lines for LILA strainmeter sensitivity
lilacolumns = ["f", "h"]
# dflilalow = pd.read_csv(LILAlow, sep=r'\s+', header=None, names=lilacolumns,
#                         index_col=False)
# dflilamedium = pd.read_csv(LILAmedium, sep=r'\s+', header=None,
#                            names=lilacolumns, index_col=False)
# dflilahigh = pd.read_csv(LILAhigh, sep=r'\s+', header=None, names=lilacolumns,
#                          index_col=False)
# print(dflilalow, dflilamedium, dflilahigh)
# plt.plot(dflilalow['f'].values, dflilalow['h'].values, color='tab:purple',
#          linestyle=":", label="LILA conservative")
# plt.plot(dflilamedium['f'].values, dflilamedium['h'].values, color='tab:purple',
#          linestyle="--", label="LILA baseline")
# plt.plot(dflilahigh['f'].values, dflilahigh['h'].values, color='tab:purple',
#          linestyle="-", label="LILA ambitious")
dflilapioneer = pd.read_csv(LILA_Pioneer,  sep=r'\s+', header=None,
                            names=lilacolumns, index_col=False, skiprows=4)
dflilahorizon = pd.read_csv(LILA_Horizon,  sep=r'\s+', header=None,
                            names=lilacolumns, index_col=False, skiprows=4)
plt.plot(dflilapioneer['f'].values, dflilapioneer['h'].values*lilafactors[0],
         color='tab:purple', linestyle='--', label='LILA Pioneer (5 km)') 
plt.plot(dflilahorizon['f'].values, dflilahorizon['h'].values*lilafactors[1],
         color='tab:purple', linestyle='-', label='LILA Horizon (40 km)')


plt.xlabel("Frequency (Hz)")
plt.ylabel("Characteristic strain")
plt.ylim((hmin, hmax))
plt.legend()
plt.savefig(fileout, dpi=300.0)
#plt.show()
           
