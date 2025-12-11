#!/usr/bin/env python3
"""
Script to record 5 videos of 2 minutes each from the live stream at https://tw.live/cam/?id=BOT243
"""

import subprocess
import re
import requests
import time
import os
from datetime import datetime


def extract_stream_url(webpage_url):
    """
    Extract the m3u8 stream URL from the webpage.
    """
    print(f"Fetching webpage: {webpage_url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(webpage_url, headers=headers, timeout=10)
        response.raise_for_status()
        html_content = response.text
        
        # Try to find m3u8 URL in the HTML
        # Common patterns: .m3u8, or embedded in JavaScript
        m3u8_patterns = [
            r'https?://[^\s"\'<>]+\.m3u8[^\s"\'<>]*',
            r'["\']([^"\']*\.m3u8[^"\']*)["\']',
            r'src["\']?\s*[:=]\s*["\']([^"\']*\.m3u8[^"\']*)["\']',
            r'url["\']?\s*[:=]\s*["\']([^"\']*\.m3u8[^"\']*)["\']',
            r'source["\']?\s*[:=]\s*["\']([^"\']*\.m3u8[^"\']*)["\']',
        ]
        
        found_urls = []
        for pattern in m3u8_patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            if matches:
                for match in matches:
                    url = match if match.startswith('http') else match
                    if '.m3u8' in url and url not in found_urls:
                        found_urls.append(url)
        
        # Test found URLs to see which one works
        # Try GET request to fetch m3u8 content and validate it
        for url in found_urls:
            print(f"Testing stream URL: {url}")
            try:
                # Try GET request to fetch the m3u8 file content
                test_response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
                if test_response.status_code == 200:
                    content = test_response.text
                    # Validate it's actually an m3u8 file
                    if content.strip().startswith('#EXTM3U') or '#EXTINF' in content:
                        print(f"✓ Found valid m3u8 stream URL: {url}")
                        return url
                    else:
                        print(f"  URL returned non-m3u8 content, skipping...")
            except Exception as e:
                # If GET fails, try HEAD as fallback
                try:
                    test_response = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
                    if test_response.status_code in [200, 302]:
                        # If HEAD works, assume it's valid (some servers don't support HEAD properly)
                        print(f"✓ Found stream URL (validated via HEAD): {url}")
                        return url
                except:
                    continue
        
        # If no m3u8 found in HTML, try to construct it from common patterns
        # For tw.live, the stream might be at a predictable location
        camera_id = re.search(r'id=([^&]+)', webpage_url)
        if camera_id:
            camera_id = camera_id.group(1)
            # Try common m3u8 URL patterns based on known formats
            potential_urls = [
                f"https://jtmctrafficcctv3.gov.taipei/NVR/{camera_id}/live.m3u8",
                f"https://tw.live/stream/{camera_id}.m3u8",
                f"https://tw.live/api/stream/{camera_id}.m3u8",
            ]
            
            for url in potential_urls:
                print(f"Trying potential URL: {url}")
                try:
                    # Try GET to fetch and validate m3u8 content
                    test_response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
                    if test_response.status_code == 200:
                        content = test_response.text
                        if content.strip().startswith('#EXTM3U') or '#EXTINF' in content:
                            print(f"✓ Found valid m3u8 stream URL: {url}")
                            return url
                except:
                    # If GET fails, try HEAD as fallback
                    try:
                        test_response = requests.head(url, headers=headers, timeout=5, allow_redirects=True)
                        if test_response.status_code in [200, 302]:
                            print(f"✓ Found accessible stream URL: {url}")
                            return url
                    except:
                        continue
        
        # If we found URLs but validation failed, try using the first one anyway
        # (sometimes servers block HEAD/GET but ffmpeg can still access the stream)
        if found_urls:
            print(f"Validation failed, but using first found URL as fallback: {found_urls[0]}")
            return found_urls[0]
        
        print("Could not find m3u8 URL in HTML. Trying alternative method...")
        return None
        
    except Exception as e:
        print(f"Error fetching webpage: {e}")
        return None


def get_stream_url_alternative(webpage_url):
    """
    Alternative method: Use yt-dlp to extract stream URL if available.
    """
    try:
        print("Attempting to extract stream URL using yt-dlp...")
        result = subprocess.run(
            ['yt-dlp', '--dump-json', '--no-playlist', webpage_url],
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            if 'url' in data:
                stream_url = data['url']
                print(f"✓ Extracted stream URL via yt-dlp: {stream_url}")
                return stream_url
            elif 'formats' in data and len(data['formats']) > 0:
                # Try to find m3u8 in formats
                for fmt in data['formats']:
                    if 'url' in fmt and '.m3u8' in fmt['url']:
                        stream_url = fmt['url']
                        print(f"✓ Extracted stream URL via yt-dlp: {stream_url}")
                        return stream_url
        else:
            print(f"yt-dlp error: {result.stderr}")
    except FileNotFoundError:
        print("yt-dlp not found. Install with: pip install yt-dlp")
    except subprocess.TimeoutExpired:
        print("yt-dlp timeout. Stream might be slow to respond.")
    except Exception as e:
        print(f"Error with yt-dlp: {e}")
    
    return None


def record_video(stream_url, output_filename, duration_seconds=120):
    """
    Record a video from the stream URL for the specified duration.
    
    Args:
        stream_url: The m3u8 stream URL
        output_filename: Output video filename
        duration_seconds: Duration to record in seconds (default 120 = 2 minutes)
    """
    print(f"\nRecording video {output_filename} for {duration_seconds} seconds...")
    
    # FFmpeg command to record the stream
    ffmpeg_cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'info',
        '-i', stream_url,
        '-t', str(duration_seconds),  # Duration in seconds
        '-c', 'copy',  # Copy codec (no re-encoding for speed)
        '-f', 'mp4',
        '-y',  # Overwrite output file if exists
        output_filename
    ]
    
    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print(f"✓ Successfully recorded: {output_filename}")
            return True
        else:
            print(f"✗ Error recording {output_filename}:")
            print(stderr)
            return False
            
    except Exception as e:
        print(f"✗ Exception while recording {output_filename}: {e}")
        return False


def main():
    """
    Main function to record 5 videos of 2 minutes each.
    """
    webpage_url = "https://tw.live/cam/?id=BOT243"
    num_videos = 30
    duration_minutes = 2
    duration_seconds = duration_minutes * 60
    
    # MANUAL STREAM URL (uncomment and set if automatic extraction fails)
    # If you know the m3u8 URL, uncomment the line below and set it:
    # manual_stream_url = "https://your-stream-url-here.m3u8"
    manual_stream_url = None
    
    print("=" * 60)
    print("Live Stream Recorder")
    print("=" * 60)
    print(f"Target: {webpage_url}")
    print(f"Recording: {num_videos} videos")
    print(f"Duration per video: {duration_minutes} minutes ({duration_seconds} seconds)")
    print("=" * 60)
    
    # Use manual stream URL if provided, otherwise extract automatically
    if manual_stream_url:
        stream_url = manual_stream_url
        print(f"Using manually specified stream URL: {stream_url}")
    else:
        # Extract stream URL automatically
        stream_url = extract_stream_url(webpage_url)
        
        if not stream_url:
            print("\nTrying alternative method to extract stream URL...")
            stream_url = get_stream_url_alternative(webpage_url)
        
        if not stream_url:
            print("\n❌ Could not extract stream URL automatically.")
            print("\nTroubleshooting steps:")
            print("   1. Check if the website is accessible in your browser")
            print("   2. Open browser developer tools (F12) and check Network tab")
            print("   3. Look for .m3u8 requests when the page loads")
            print("   4. Install yt-dlp for better extraction: pip install yt-dlp")
            print("\nAlternative: Edit this script and set manual_stream_url above.")
            return
    
    # Create output directory
    output_dir = "recordings"
    os.makedirs(output_dir, exist_ok=True)
    
    # Record 5 videos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i in range(1, num_videos + 1):
        output_filename = os.path.join(output_dir, f"recording_{timestamp}_part{i:02d}.mp4")
        
        success = record_video(stream_url, output_filename, duration_seconds)
        
        if not success:
            print(f"\n⚠ Warning: Failed to record video {i}. Continuing with next video...")
        
        # Small delay between recordings (optional)
        if i < num_videos:
            print(f"\nWaiting 2 seconds before next recording...")
            time.sleep(2)
    
    print("\n" + "=" * 60)
    print("Recording complete!")
    print(f"Videos saved in: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()

