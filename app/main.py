import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import uvicorn
from typing import List
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from core.config import settings

GUITAR_STRINGS = {
    "E": 82.41,  # Low E (E2)
    "A": 110.00, # A2
    "D": 146.83, # D3
    "G": 196.00, # G3
    "B": 246.94, # B3
    "e": 329.63  # High E (E4)
}

app = FastAPI(title="Guitar Tuner API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a directory for static files if it doesn't exist
os.makedirs("static", exist_ok=True)



# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# Audio processing class
class AudioProcessor:
    def __init__(self):
        self.buffer_size = 8192  # Larger buffer for better low-frequency resolution
        self.buffer = np.zeros(self.buffer_size)
        self.buffer_index = 0
        self.sample_rate = 44100  # Default, will be updated from client
        self.target_string = None
        self.target_frequency = None
        # Keep track of recent frequency measurements for smoothing
        self.recent_frequencies = []
        self.max_recent = 5  # Number of recent values to keep

    def set_target(self, string_name, frequency):
        self.target_string = string_name
        self.target_frequency = frequency
        print(f"Target set to {string_name}: {frequency} Hz")

    def add_samples(self, samples, sample_rate):
        self.sample_rate = sample_rate
        # Normalize samples from 0-255 to -1 to 1
        normalized_samples = (np.array(samples) / 128.0) - 1.0
        
        # Add to buffer with circular wrapping
        samples_to_add = len(normalized_samples)
        space_available = self.buffer_size - self.buffer_index
        
        if samples_to_add <= space_available:
            # Fits in remaining space
            self.buffer[self.buffer_index:self.buffer_index+samples_to_add] = normalized_samples
            self.buffer_index += samples_to_add
        else:
            # Fill remaining space and wrap around
            self.buffer[self.buffer_index:] = normalized_samples[:space_available]
            remaining = samples_to_add - space_available
            self.buffer[:remaining] = normalized_samples[space_available:]
            self.buffer_index = remaining
            
        # If buffer is full or nearly full, process it
        if self.buffer_index >= self.buffer_size * 0.75:
            frequency = self.process_buffer()
            
            # If we got a valid frequency, add it to recent measurements
            if frequency is not None:
                self.recent_frequencies.append(frequency)
                self.recent_frequencies = self.recent_frequencies[-self.max_recent:]  # Keep only most recent
                
                # Return the median of recent frequencies for stability
                if len(self.recent_frequencies) >= 3:
                    return np.median(self.recent_frequencies)
                else:
                    return frequency
            
            return None
        return None

    def process_buffer(self):
        # Apply bandpass filter to focus on the guitar frequency range
        nyquist = 0.5 * self.sample_rate
        low = 70.0 / nyquist
        high = 400.0 / nyquist
        b, a = butter(2, [low, high], btype='band')
        filtered_buffer = filtfilt(b, a, self.buffer)
        
        # Apply window function to reduce spectral leakage
        windowed_buffer = filtered_buffer * np.hanning(self.buffer_size)
        
        # Zero padding for better frequency resolution
        padded_buffer = np.zeros(self.buffer_size * 2)
        padded_buffer[:self.buffer_size] = windowed_buffer
        
        # Compute FFT
        fft_result = np.fft.rfft(padded_buffer)
        fft_magnitude = np.abs(fft_result)
        
        # Find frequency bins
        freq_bins = np.fft.rfftfreq(self.buffer_size * 2, 1.0/self.sample_rate)
        
        # Find peaks in the spectrum
        peaks, properties = find_peaks(fft_magnitude, height=np.max(fft_magnitude)/20, distance=5)
        
        if len(peaks) > 0:
            # Get peak frequencies and their magnitudes
            peak_freqs = freq_bins[peaks]
            peak_mags = fft_magnitude[peaks]
            
            # Filter peaks to guitar range (expand a bit to catch fundamentals)
            valid_peaks = []
            valid_mags = []
            
            for i, freq in enumerate(peak_freqs):
                if 70 <= freq <= 400:  # Expanded range to catch some higher fundamentals
                    valid_peaks.append(freq)
                    valid_mags.append(peak_mags[i])
            
            if not valid_peaks:
                return None
                
            # If we have a target frequency, look for peaks near it
            if self.target_frequency:
                # Define frequency ranges for each string
                # For standard tuning E2 (82.41) to E4 (329.63)
                string_ranges = {
                    "E": (75, 90),    # Low E
                    "A": (100, 120),  # A
                    "D": (135, 158),  # D
                    "G": (185, 208),  # G
                    "B": (235, 260),  # B
                    "e": (310, 350)   # High E
                }
                
                if self.target_string in string_ranges:
                    min_freq, max_freq = string_ranges[self.target_string]
                    
                    # Look for peaks in the expected range for this string
                    closest_peaks = []
                    closest_mags = []
                    
                    for i, freq in enumerate(valid_peaks):
                        if min_freq <= freq <= max_freq:
                            closest_peaks.append(freq)
                            closest_mags.append(valid_mags[i])
                    
                    # If we found peaks in the target range, use the strongest one
                    if closest_peaks:
                        strongest_idx = np.argmax(closest_mags)
                        return closest_peaks[strongest_idx]
            
            # If we didn't return a targeted frequency, use autocorrelation to find fundamental
            # This is a better approach for finding the fundamental frequency
            
            # Compute autocorrelation via inverse FFT of power spectrum
            power_spectrum = np.abs(fft_result)**2
            autocorr = np.fft.irfft(power_spectrum)
            
            # Normalize
            autocorr = autocorr / autocorr[0]
            
            # Find peaks in autocorrelation (excluding the first peak at lag 0)
            ac_peaks, _ = find_peaks(autocorr[1:], height=0.2)
            ac_peaks = ac_peaks + 1  # Adjust for the offset from excluding first point
            
            if len(ac_peaks) > 0:
                # The first prominent peak corresponds to the period
                # Convert lag to frequency
                period = ac_peaks[0]
                fundamental = self.sample_rate / period
                
                # Check if this fundamental is in the guitar range
                if 70 <= fundamental <= 350:
                    return fundamental
            
            # If autocorrelation didn't work, fall back to our original approach
            fundamental_candidates = []
            
            # Sort by magnitude (descending)
            sorted_indices = np.argsort(valid_mags)[::-1]
            sorted_peaks = [valid_peaks[i] for i in sorted_indices]
            
            for freq in sorted_peaks[:3]:  # Look at top 3 strongest peaks
                # Check if this could be a harmonic (multiple) of a lower frequency
                is_harmonic = False
                for div in [2, 3, 4]:  # Check if it's a 2nd, 3rd or 4th harmonic
                    fundamental = freq / div
                    # If potential fundamental is in guitar range
                    if 75 <= fundamental <= 350:
                        # Look for this fundamental or its nearby frequency in peaks
                        for other_freq in valid_peaks:
                            if abs(other_freq - fundamental) < 5:  # Within 5 Hz
                                fundamental_candidates.append(fundamental)
                                is_harmonic = True
                                break
                
                # If not a harmonic, it might be a fundamental itself
                if not is_harmonic:
                    fundamental_candidates.append(freq)
            
            # Return the most likely fundamental frequency
            if fundamental_candidates:
                # If we have candidates, return the lowest one (most likely to be fundamental)
                return min(fundamental_candidates)
            elif valid_peaks:
                # As a fallback, return the strongest peak in the valid range
                return valid_peaks[np.argmax(valid_mags)]
        
        return None

audio_processor = AudioProcessor()

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("static/index.html", "r") as f:
        return f.read()

@app.websocket("/ws/tuner")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["action"] == "start":
                # Set target string and frequency
                audio_processor.set_target(
                    message["string"], 
                    message["frequency"]
                )
                
            elif message["action"] == "stop":
                # Reset processor if needed
                pass
                
            elif message["action"] == "set_string":
                # Update target string and frequency
                audio_processor.set_target(
                    message["string"], 
                    message["frequency"]
                )
                
            elif message["action"] == "process_audio":
                # Process incoming audio data
                audio_data = message["data"]
                sample_rate = message["sampleRate"]
                
                frequency = audio_processor.add_samples(audio_data, sample_rate)
                
                if frequency is not None:
                    # Send detected frequency back to client
                    await manager.send_personal_message(
                        json.dumps({"frequency": frequency}),
                        websocket
                    )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=settings.debug)