"""
Plot up acceleration of likely lunar seismicity, highlight mode frequencies and
compare with noise of seismometers
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, next_fast_len
from matplotlib import mlab
import math
from obspy import Stream, Trace, UTCDateTime

fileout = "acc_figure.png"
modes_root = "./modes_300mHz"
mdl = "weber2010hr"
DMQ_path = "800km_3hrs"
SMQ_path = "30km_3hrs"
DMQ_45d_path = "800km_45deg_3hrs"


# Plot variables
amin = 1.e-19
amax = 1.e-6

# Quake variables
DMQ_moment = 1.e13
DMQ_max_moment = math.pow(10, 13.8)
DMQ_max_scale = DMQ_max_moment/DMQ_moment
SMQ_moment = 3.98e16
SMQ_scale = 1.e-2 # To account for me mistakenly equating mb to Mw

seismometer_path = "./Seismometers"
seismometers = ["FSS", "LOVBB", "HarmsOptomechanical", "HarmsCryomagnetic",
                "NiobWatts", "SiWatts"]
seislabels = ["FSS", "LOVBB", "LGWA1", "LGWA2", "LGWA3", "LGWA4"]
seiscolors = ["tab:red", "tab:purple", "lightgray", "darkgray", "gray", "black"]

distances = [5.0, 5.016, 5.033, 20.0, 20.016, 20.033, 40.0, 40.016, 40.033,
             90.0, 90.016, 90.033]
# This is zero-indexed to match the distance array, but the station names
# output by nms are 1-indexed
stn = 6

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

# Read in the data and differentiate to acceleration
stdmq = Stream()

for comp in ['Z', 'N', 'E']:
    filename = "{}/{}/U{}_{:04d}".format(modes_root, DMQ_path, comp, stn+1)
    df = pd.read_fwf(filename, header=None, names=['Time', comp])
    delta = df['Time'].values[1]-df['Time'].values[0]
    tr = Trace(data=df[comp].values)
    tr.stats['delta'] = delta
    tr.stats['channel'] = "MH{}".format(comp)
    stnname = "DMQ{:02d}".format(stn+1)
    tr.stats['network'] = network
    tr.stats['station'] = stnname
    tr.stats['starttime'] = reftime + df['Time'].values[0]
    # differentiate twice to acceleration
    # tr.differentiate()
    # tr.differentiate()
    stdmq = stdmq + tr

#Low pass filter to limit some ringing
fcorner = 0.95*fmax
stdmq.filter('lowpass', freq=fcorner)
if iftrim:
    stime = reftime
    etime = stime+trimlength
    stdmq.trim(starttime=stime, endtime=etime)

# Write them to mseed files
filename = "{}/fmax{:04d}_{}_DMQ_acc.mseed".format(mseeds_path, int(fmax*1000),
                                                   stdmq[0].stats['station'])
stdmq.write(filename)

# Scale typical DMQ up to the max observed DMQ moment
stdmqmax = stdmq.copy()
for tr in stdmqmax:
    tr.data = DMQ_max_scale*tr.data

# Read in the data for off-axis and differentiate to acceleration
stdmq_off = Stream()

for comp in ['Z', 'N', 'E']:
    filename = "{}/{}/U{}_{:04d}".format(modes_root, DMQ_45d_path, comp, stn+1)
    df = pd.read_fwf(filename, header=None, names=['Time', comp])
    delta = df['Time'].values[1]-df['Time'].values[0]
    tr = Trace(data=df[comp].values)
    tr.stats['delta'] = delta
    tr.stats['channel'] = "MH{}".format(comp)
    stnname = "DMQ{:02d}".format(stn+1)
    tr.stats['network'] = network
    tr.stats['station'] = stnname
    tr.stats['starttime'] = reftime + df['Time'].values[0]
    # differentiate twice to acceleration
    # tr.differentiate()
    # tr.differentiate()
    stdmq_off = stdmq_off + tr

#Low pass filter to limit some ringing
fcorner = 0.95*fmax
stdmq_off.filter('lowpass', freq=fcorner)
if iftrim:
    stime = reftime
    etime = stime+trimlength
    stdmq_off.trim(starttime=stime, endtime=etime)

# Write them to mseed files
filename = "{}/fmax{:04d}_{}_DMQ_off_acc.mseed".format(mseeds_path, int(fmax*1000),
                                                       stdmq_off[0].stats['station'])
stdmq_off.write(filename)

# Do the same for SMQ
stsmq = Stream()

for comp in ['Z', 'N', 'E']:
    filename = "{}/{}/U{}_{:04d}".format(modes_root, SMQ_path, comp, stn+1)
    df = pd.read_fwf(filename, header=None, names=['Time', comp])
    delta = df['Time'].values[1]-df['Time'].values[0]
    tr = Trace(data=df[comp].values*SMQ_scale)
    tr.stats['delta'] = delta
    tr.stats['channel'] = "MH{}".format(comp)
    stnname = "SMQ{:02d}".format(stn+1)
    tr.stats['network'] = network
    tr.stats['station'] = stnname
    tr.stats['starttime'] = reftime + df['Time'].values[0]
    # differentiate twice to acceleration
    # tr.differentiate()
    # tr.differentiate()
    stsmq = stsmq + tr

#Low pass filter to limit some ringing
fcorner = 0.95*fmax
stsmq.filter('lowpass', freq=fcorner)
if iftrim:
    stime = reftime
    etime = stime+trimlength
    stsmq.trim(starttime=stime, endtime=etime)

# Write to mseed file
filename = "{}/fmax{:04d}_{}_SMQ_acc.mseed".format(mseeds_path, int(fmax*1000),
                                                   stsmq[0].stats['station'])
stsmq.write(filename)

# Read in mode data and get mode frequencies below 10 mHz
modefile = "{}/{}S.csv".format(modes_root, mdl)
modedf = pd.read_csv(modefile, index_col=False)
maxf = 3
modedf_low = modedf.loc[(modedf['f'] < maxf) & (modedf['l'] > 0)]
modefreqs = modedf_low['f'].values * 1.e-3 # convert to Hz

# Plot up mlab and fft estimates (eventually only one, but this is just to
# verify calculation). All using Z component now
fs = stdmq[0].stats['sampling_rate']
# (stdmqPxx, stdmqfreqs) = mlab.psd(stdmq[0].data, Fs=fs, NFFT=nfft,
#                                   noverlap=noverlap, detrend='linear')
# plt.loglog(stdmqfreqs, np.sqrt(stdmqPxx), label="DMQ mlab")
stdmq[0].detrend('demean')
npts = stdmq[0].stats['npts']
nfft_scipy = 128*next_fast_len(npts)
scale = 1./np.sqrt(npts)
pos_freq = (nfft_scipy + 1 ) // 2
spec = fft(stdmq[0].data, nfft_scipy)[:pos_freq]
freq = fftfreq(nfft_scipy, 1 / fs)[:pos_freq]
# Differentiate to acceleration in frequency domain
omega = 2.*math.pi*freq
omegasq = np.multiply(omega, omega)
spec = np.multiply(omegasq, spec)
plt.loglog(freq, scale*np.abs(spec), label="DMQ (typical)")

fs = stdmqmax[0].stats['sampling_rate']
# (stdmqmaxPxx, stdmqmaxfreqs) = mlab.psd(stdmq[0].data, Fs=fs, NFFT=nfft,
#                                         noverlap=noverlap, detrend='linear')
# plt.loglog(stdmqmaxfreqs, np.sqrt(stdmqmaxPxx), label="DMQ largest mlab")
stdmqmax[0].detrend('demean')
npts = stdmqmax[0].stats['npts']
nfft_scipy = 128*next_fast_len(npts)
scale = 1./np.sqrt(npts)
pos_freq = (nfft_scipy + 1 ) // 2
spec = fft(stdmqmax[0].data, nfft_scipy)[:pos_freq]
freq = fftfreq(nfft_scipy, 1 / fs)[:pos_freq]
# Differentiate to acceleration in frequency domain
omega = 2.*math.pi*freq
omegasq = np.multiply(omega, omega)
spec = np.multiply(omegasq, spec)
plt.loglog(freq, scale*np.abs(spec), label="DMQ (largest)")

fs = stsmq[0].stats['sampling_rate']
# (stsmqPxx, stsmqfreqs) = mlab.psd(stsmq[0].data, Fs=fs, NFFT=nfft,
#                                   noverlap=noverlap, detrend='linear')
# plt.loglog(stsmqfreqs, np.sqrt(stsmqPxx), label="SMQ mlab")
stsmq[0].detrend('demean')
npts = stsmq[0].stats['npts']
nfft_scipy = 128*next_fast_len(npts)
scale = 1./np.sqrt(npts)
pos_freq = (nfft_scipy + 1 ) // 2
spec = fft(stsmq[0].data, nfft_scipy)[:pos_freq]
freq = fftfreq(nfft_scipy, 1 / fs)[:pos_freq]
# Differentiate to acceleration in frequency domain
omega = 2.*math.pi*freq
omegasq = np.multiply(omega, omega)
spec = np.multiply(omegasq, spec)
plt.loglog(freq, scale*np.abs(spec), label="SMQ (largest)")

fs = stdmq_off[0].stats['sampling_rate']
# (stsmqPxx, stsmqfreqs) = mlab.psd(stsmq[0].data, Fs=fs, NFFT=nfft,
#                                   noverlap=noverlap, detrend='linear')
# plt.loglog(stsmqfreqs, np.sqrt(stsmqPxx), label="SMQ mlab")
stdmq_off[0].detrend('demean')
npts = stdmq_off[0].stats['npts']
nfft_scipy = 128*next_fast_len(npts)
scale = 1./np.sqrt(npts)
pos_freq = (nfft_scipy + 1 ) // 2
spec = fft(stdmq_off[0].data, nfft_scipy)[:pos_freq]
freq = fftfreq(nfft_scipy, 1 / fs)[:pos_freq]
# Differentiate to acceleration in frequency domain
omega = 2.*math.pi*freq
omegasq = np.multiply(omega, omega)
spec = np.multiply(omegasq, spec)
plt.loglog(freq, scale*np.abs(spec), label="DMQ (off-axis)", color="tab:blue", linestyle="--")

# add in lines for mode frequencies
plt.axvline(x=modefreqs[0], color='r', linestyle='--', linewidth=0.5,
            label="Modes (<{} mHz)".format(maxf))
for f in modefreqs[1:]:
    plt.axvline(x=f, color='r', linestyle='--', linewidth=0.5)

plt.xlim((1.e-5, 0.5))

# Plot noise curves
for i, seismometer in enumerate(seismometers):
    filename = "{}/{}".format(seismometer_path, seismometer)
    df = pd.read_fwf(filename, header=None, names=['f', 'acc'])
    plt.plot(df['f'].values, df['acc'].values, label=seislabels[i],
             color=seiscolors[i])

plt.xlabel(r"Frequency ($Hz$)")
plt.ylabel(r"Accel Spectral Density ($m/s^2/\sqrt{Hz}$)")
plt.ylim((amin, amax))
plt.legend()
plt.savefig(fileout)

