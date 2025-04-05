
from gtts import gTTS
from moviepy import AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ImageClip
import textwrap

open_ai_api_key = "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"
image_directory = "/home/gmohit/Pictures/images"
hi_text = """भारत सरकार ने 1 फरवरी 2025 को वित्त वर्ष 2025-26 के लिए बजट प्रस्तुत किया, जिसमें मध्यम वर्ग की क्रय शक्ति बढ़ाने, समावेशी विकास को प्रोत्साहित करने और निजी निवेश को बढ़ावा देने पर विशेष ध्यान दिया गया है। बजट के प्रमुख बिंदु निम्नलिखित हैं:

आयकर में राहत:

नया कर स्लैब:
1.2 मिलियन रुपये तक की वार्षिक आय पर कोई कर नहीं।
2.4 मिलियन रुपये से अधिक की आय पर अधिकतम 30% कर दर लागू होगी।
इससे मध्यम वर्ग की क्रय शक्ति बढ़ेगी और उपभोक्ता मांग में वृद्धि की उम्मीद है।
कृषि क्षेत्र में सुधार:

दालों और कपास के लिए विशेष मिशन शुरू किए जाएंगे ताकि उत्पादन में वृद्धि हो सके।
17 मिलियन किसानों को लक्षित करते हुए उच्च उपज वाली फसल कार्यक्रम की शुरुआत की जाएगी।
किसानों के लिए सब्सिडी युक्त ऋण की सीमा बढ़ाई जाएगी।
उद्योग और निवेश:

बीमा क्षेत्र में प्रत्यक्ष विदेशी निवेश (FDI) की सीमा बढ़ाकर 100% की जाएगी।
स्टार्टअप्स, लघु उद्यमों और विनिर्माण को समर्थन देने के लिए प्रोत्साहन और फंड की व्यवस्था की जाएगी।
राष्ट्रीय विनिर्माण मिशन की स्थापना की जाएगी।
इन्फ्रास्ट्रक्चर और ऊर्जा:

इन्फ्रास्ट्रक्चर विकास और क्षेत्रीय हवाई संपर्क को बढ़ावा दिया जाएगा।
महत्वपूर्ण खनिजों के विकास पर ध्यान दिया जाएगा।
न्यूक्लियर एनर्जी मिशन के तहत 2047 तक 100 गीगावाट परमाणु ऊर्जा उत्पादन का लक्ष्य रखा गया है।
आर्थिक अनुमान:

सकल घरेलू उत्पाद (GDP) में 10.1% की नाममात्र वृद्धि का अनुमान है।
वित्तीय घाटा GDP का 4.4% रहने की उम्मीद है।
कुल बजट व्यय 50.65 ट्रिलियन रुपये होने का अनुमान है।
इस बजट के माध्यम से सरकार का उद्देश्य आर्थिक विकास को गति देना, मध्यम वर्ग को राहत प्रदान करना और विभिन्न क्षेत्रों में सुधार लाना है।

"""


font_path = "/usr/share/fonts/truetype/samyak/Samyak-Devanagari.ttf"

def get_tts(text):
    # === Step 1: Convert Text to Speech ===
    audio_path = "../audio.mp3"
    tts = gTTS(text, lang="hi")  # Use Hindi language
    tts.save(audio_path)

    # Load generated audio
    audio = AudioFileClip(audio_path)
    return audio

def generate_subtitles(text, duration, max_chars=30):
    words = text.split()
    subtitles = []
    start_time = 0
    time_per_word = duration / len(words)

    for i in range(0, len(words), max_chars):
        subtitle_text = " ".join(words[i:i+max_chars])
        end_time = start_time + len(words[i:i+max_chars]) * time_per_word
        subtitles.append((subtitle_text, start_time, end_time))
        start_time = end_time

    #return subtitles

    #subtitles = generate_subtitles(text, audio_duration)

    # 🔹 Step 5: Create Subtitle Clips
    subtitle_clips = []
    for subtitle, start, end in subtitles:
        text_clip = TextClip(subtitle, fontsize=40, font="DejaVu-Sans", color="white", bg_color="black", size=(1200, 100))
        text_clip = text_clip.set_position(("center", "bottom")).set_start(start).set_end(end)
        subtitle_clips.append(text_clip)
    return subtitle_clips



def get_video(audio, image_files):
    num_images = len(image_files)
    wrapped_texts = textwrap.wrap(hi_text, width=50)  # Adjust 'width' as needed
    text_chunks = wrapped_texts[:num_images]  # Trim to fit available images
    # Calculate duration per image based on audio length
    duration_per_image = audio.duration / num_images
    #durations = [duration_per_image] * num_images  # Equal duration for all images

    # === Step 3: Create Video with Images and Overlay Text ===
    clips = []
    for img in image_files:
        #img_clip = ImageSequenceClip([img], durations=[duration_per_image])
        img_clip = ImageClip(img)
        img_clip.duration = duration_per_image
        # text_clip = TextClip(font=font_path, font_size=20, text=text_chunk, color="orange", bg_color="black")#, size=(1280, 100))
        # text_clip.duration = duration_per_image
        # text_clip.with_position("center", "bottom")

        # Merge image and text
        #final_clip = CompositeVideoClip([img_clip])
        #final_clip.duration = duration_per_image
        clips.append(img_clip)

    # Concatenate all clips
    #final_video = ImageSequenceClip(clips, durations=durations)
    final_video = concatenate_videoclips(clips, method="compose")
    final_video.audio = audio

    # Save the final video
    output_video = f"{image_directory}/output.mp4"
    final_video.write_videofile(output_video, fps=24, codec="libx264", audio_codec="aac")

    print("✅ Video created successfully: output.mp4")


if __name__ == "__main__":
    image_files = [f"{image_directory}/1.jpg",
                   f"{image_directory}/2.jpg",
                   f"{image_directory}/3.jpg",
                   f"{image_directory}/4.jpg",
                   f"{image_directory}/5.jpg",
                   f"{image_directory}/6.jpg",
                   f"{image_directory}/7.jpg"
                   ]

    audio = get_tts(hi_text)
    get_video(audio, image_files)
