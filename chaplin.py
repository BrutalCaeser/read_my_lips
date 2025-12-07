import cv2
import time
from ollama import AsyncClient
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import os
from pynput import keyboard
import asyncio


class ChaplinOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class Chaplin:
    def __init__(self):
        self.vsr_model = None

        # flag to toggle recording
        self.recording = False

        # thread stuff
        self.executor = ThreadPoolExecutor(max_workers=1)

        # video params
        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25

        # setup keyboard controller for typing
        self.kbd_controller = keyboard.Controller()

        # setup async ollama client
        self.ollama_client = AsyncClient()

        # setup asyncio event loop in background thread
        self.loop = asyncio.new_event_loop()
        self.async_thread = ThreadPoolExecutor(max_workers=1)
        self.async_thread.submit(self._run_event_loop)

        # sequence tracking to ensure outputs are typed in order
        self.next_sequence_to_type = 0
        self.current_sequence = 0  # counter for assigning sequence numbers
        self.typing_lock = None  # will be created in async loop
        self._init_async_resources()

        # setup global hotkey for toggling recording with option/alt key
        self.hotkey = keyboard.GlobalHotKeys({
            '<alt>': self.toggle_recording
        })
        self.hotkey.start()

    def _run_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _init_async_resources(self):
        """Initialize async resources in the async loop"""
        future = asyncio.run_coroutine_threadsafe(
            self._create_async_lock(), self.loop)
        future.result()  # wait for it to complete

    async def _create_async_lock(self):
        """Create asyncio.Lock and Condition in the event loop's context"""
        self.typing_lock = asyncio.Lock()
        self.typing_condition = asyncio.Condition(self.typing_lock)

    def toggle_recording(self):
        # toggle recording when alt/option key is pressed
        self.recording = not self.recording

    async def correct_output_async(self, output, sequence_num):
        # Debugging: Start of correct_output_async
        print("Debug: Entering correct_output_async...")

        # perform inference on the raw output to get back a "correct" version
        # Debugging: Before calling Ollama client
        print("Debug: About to call Ollama client with timeout...")

        try:
            # Changing the model to a different one for testing
            response = await asyncio.wait_for(
                self.ollama_client.chat(
                    model='qwen3:0.6b',  # Replace with the desired model name
                    messages=[
                        {
                            'role': 'system',
                            'content': (
                                "You are a precise text correction assistant. Your ONLY job is to fix transcription errors in lip-read text.\n"
                                "The input is raw, uppercase text from a lip-reading model. It may have wrong words that look similar on lips (e.g. 'ME' instead of 'BE').\n\n"
                                "RULES:\n"
                                "1. Fix nonsensical words based on context. Example: 'THANKS TO ME' -> 'THANKS TO BE' or just 'THANKS'.\n"
                                "2. Output normal English sentence case (e.g. 'Hello world', not 'Hello World').\n"
                                "3. Add proper punctuation at the end.\n"
                                "4. DO NOT add extra words. DO NOT repeat words.\n"
                                "5. If the sentence is mostly correct, just fix the case and punctuation.\n\n"
                                "Return JSON with 'list_of_changes' and 'corrected_text'."
                            )
                        },
                        {
                            'role': 'user',
                            'content': f"Transcription:\n\n{output}"
                        }
                    ],
                    format=ChaplinOutput.model_json_schema()
                ),
                timeout=60  # Timeout set to 60 seconds
            )

            # Debugging: Response received
            print("Debug: Response successfully received from Ollama client.")
        except asyncio.TimeoutError:
            # Debugging: Timeout occurred
            print("Debug: Timeout occurred while waiting for Ollama client response.")
            return "Timeout occurred while processing the request."
        except Exception as e:
            # Debugging: Exception occurred
            print(f"Debug: Exception occurred while calling Ollama client: {e}")
            raise

        # get only the corrected text
        chat_output = ChaplinOutput.model_validate_json(
            response['message']['content']
        )

        # Debugging: Corrected text extracted
        print(f"Debug: Corrected text extracted: {chat_output.corrected_text}")

        # if last character isn't a sentence ending (happens sometimes), add a period
        chat_output.corrected_text = chat_output.corrected_text.strip()
        if chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'

        # add space at the end
        chat_output.corrected_text += ' '

        # wait until it's this task's turn to type
        async with self.typing_condition:
            while self.next_sequence_to_type != sequence_num:
                await self.typing_condition.wait()

            # this task's turn to type the corrected text
            self.kbd_controller.type(chat_output.corrected_text)

            # increment sequence and notify next task
            self.next_sequence_to_type += 1
            self.typing_condition.notify_all()

        return chat_output.corrected_text

    def perform_inference(self, video_path):
        # perform inference on the video with the vsr model
        output = self.vsr_model(video_path)

        # print the raw output to console
        print(f"\n\033[48;5;21m\033[97m\033[1m RAW OUTPUT \033[0m: {output}\n")

        # assign sequence number for this task
        sequence_num = self.current_sequence
        self.current_sequence += 1

        # Debugging: Ensure corrected output is being processed
        print("Debug: Starting LLM correction...")
        corrected_output = asyncio.run_coroutine_threadsafe(
            self.correct_output_async(output, sequence_num),
            self.loop
        ).result()
        print("Debug: LLM correction completed.")

        # Ensure corrected text is logged explicitly
        print(f"\n\033[48;5;22m\033[97m\033[1m CORRECTED OUTPUT \033[0m: {corrected_output}\n")

        # Convert corrected text to audio
        audio_file_path = "corrected_output_audio.wav"
        from pipelines.pipeline import InferencePipeline
        InferencePipeline.text_to_speech(corrected_output, audio_file_path)
        print(f"Audio file generated: {audio_file_path}")

        # Generate talking head video
        try:
            import os
            # Enable MPS fallback for missing operators like linalg_qr
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            from float_module.generate import InferenceAgent, InferenceOptions
            
            print("\n\033[48;5;22m\033[97m\033[1m GENERATING TALKING HEAD VIDEO \033[0m\n")
            
            # Setup options
            # Pass empty list to avoid conflict with hydra arguments
            opt = InferenceOptions().parse(args=[])
            opt.ckpt_path = os.path.abspath("float_module/checkpoints/float.pth")
            opt.wav2vec_model_path = os.path.abspath("float_module/checkpoints/wav2vec2-base-960h")
            opt.audio2emotion_path = os.path.abspath("float_module/checkpoints/wav2vec-english-speech-emotion-recognition")
            
            # Use MPS if available, else CPU
            import torch
            if torch.backends.mps.is_available():
                opt.rank = torch.device("mps")
            else:
                opt.rank = torch.device("cpu")
            
            # Initialize agent
            agent = InferenceAgent(opt)
            
            # Set paths
            ref_path = os.path.abspath("float_module/prof_nadim.png") # Default reference image
            aud_path = os.path.abspath(audio_file_path)
            res_video_path = os.path.abspath(f"output_video_{sequence_num}.mp4")
            
            # Run inference
            agent.run_inference(
                res_video_path,
                ref_path,
                aud_path,
                a_cfg_scale=opt.a_cfg_scale,
                r_cfg_scale=opt.r_cfg_scale,
                e_cfg_scale=opt.e_cfg_scale,
                emo=opt.emo,
                nfe=opt.nfe,
                no_crop=opt.no_crop,
                seed=opt.seed
            )
            print(f"\n\033[48;5;22m\033[97m\033[1m VIDEO GENERATED: {res_video_path} \033[0m\n")
            
            # Auto-play the generated video
            try:
                import subprocess
                import platform
                
                print("Auto-playing generated video...")
                if platform.system() == 'Darwin':       # macOS
                    subprocess.call(('open', res_video_path))
                elif platform.system() == 'Windows':    # Windows
                    os.startfile(res_video_path)
                else:                                   # linux variants
                    subprocess.call(('xdg-open', res_video_path))
            except Exception as e:
                print(f"Could not auto-play video: {e}")

        except Exception as e:
            print(f"Error generating talking head video: {e}")
            import traceback
            traceback.print_exc()

        # Return corrected output and audio file path for further use
        return {
            "output": output,
            "corrected_output": corrected_output,
            "audio_file_path": audio_file_path,
            "video_path": video_path
        }

    def start_webcam(self):
        # init webcam
        cap = cv2.VideoCapture(0)

        # set webcam resolution, and get frame dimensions
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        last_frame_time = time.time()

        futures = []
        output_path = ""
        out = None
        frame_count = 0

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # remove any remaining videos that were saved to disk
                for file in os.listdir():
                    if file.startswith(self.output_prefix) and file.endswith('.mp4'):
                        os.remove(file)
                break

            current_time = time.time()

            # conditional ensures that the video is recorded at the correct frame rate
            if current_time - last_frame_time >= self.frame_interval:
                ret, frame = cap.read()
                if ret:
                    # frame compression
                    encode_param = [
                        int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    compressed_frame = cv2.imdecode(
                        buffer, cv2.IMREAD_GRAYSCALE)

                    if self.recording:
                        if out is None:
                            output_path = self.output_prefix + \
                                str(time.time_ns() // 1_000_000) + '.mp4'
                            out = cv2.VideoWriter(
                                output_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                self.fps,
                                (frame_width, frame_height),
                                False  # isColor
                            )

                        out.write(compressed_frame)

                        last_frame_time = current_time

                        # circle to indicate recording, only appears in the window and is not present in video saved to disk
                        cv2.circle(compressed_frame, (frame_width -
                                                      20, 20), 10, (0, 0, 0), -1)

                        frame_count += 1
                    # check if not recording AND video is at least 2 seconds long
                    elif not self.recording and frame_count > 0:
                        if out is not None:
                            out.release()

                        # only run inference if the video is at least 2 seconds long
                        if frame_count >= self.fps * 2:
                            futures.append(self.executor.submit(
                                self.perform_inference, output_path))
                        else:
                            os.remove(output_path)

                        output_path = self.output_prefix + \
                            str(time.time_ns() // 1_000_000) + '.mp4'
                        out = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            self.fps,
                            (frame_width, frame_height),
                            False  # isColor
                        )

                        frame_count = 0

                    # display the frame in the window
                    cv2.imshow('Chaplin', cv2.flip(compressed_frame, 1))

            # ensures that videos are handled in the order they were recorded
            for fut in futures:
                if fut.done():
                    result = fut.result()
                    # once done processing, delete the video with the video path
                    os.remove(result["video_path"])
                    futures.remove(fut)
                else:
                    break

        # release everything
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        # stop global hotkey listener
        self.hotkey.stop()

        # stop async event loop
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.async_thread.shutdown(wait=True)

        # shutdown executor
        self.executor.shutdown(wait=True)


# Ensure the test function is accessible at the module level
def test_ollama_client():
    """Test the Ollama client with a simple input to verify functionality."""
    import asyncio

    async def test():
        client = AsyncClient()
        try:
            print("Testing Ollama client...")
            response = await asyncio.wait_for(
                client.chat(
                    model='qwen3:4b',
                    messages=[
                        {
                            'role': 'system',
                            'content': "You are a test assistant."
                        },
                        {
                            'role': 'user',
                            'content': "Hello, can you respond to this simple test?"
                        }
                    ]
                ),
                timeout=30
            )
            print("Test response received:", response)
        except asyncio.TimeoutError:
            print("Test failed: Timeout occurred.")
        except Exception as e:
            print(f"Test failed: Exception occurred: {e}")

    asyncio.run(test())
