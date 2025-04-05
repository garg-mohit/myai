from moviepy import AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ImageClip
from gtts import gTTS

# 1️⃣ Define the text for subtitles and audio
text = """भारत सरकार ने 1 फरवरी 2025 को वित्त वर्ष 2025-26 के लिए बजट प्रस्तुत किया, जिसमें मध्यम वर्ग की क्रय शक्ति बढ़ाने, समावेशी विकास को प्रोत्साहित करने और निजी निवेश को बढ़ावा देने पर विशेष ध्यान दिया गया है। बजट के प्रमुख बिंदु निम्नलिखित हैं: आयकर में राहत: नया कर स्लैब: 1.2 मिलियन रुपये तक की वार्षिक आय पर कोई कर नहीं। 2.4 मिलियन रुपये से अधिक की आय पर अधिकतम 30% कर दर लागू होगी। इससे मध्यम वर्ग की क्रय शक्ति बढ़ेगी और उपभोक्ता मांग में वृद्धि की उम्मीद है। कृषि क्षेत्र में सुधार: दालों और कपास के लिए विशेष मिशन शुरू किए जाएंगे ताकि उत्पादन में वृद्धि हो सके। 17 मिलियन किसानों को लक्षित करते हुए उच्च उपज वाली फसल कार्यक्रम की शुरुआत की जाएगी। किसानों के लिए सब्सिडी युक्त ऋण की सीमा बढ़ाई जाएगी। उद्योग और निवेश: बीमा क्षेत्र में प्रत्यक्ष विदेशी निवेश (FDI) की सीमा बढ़ाकर 100% की जाएगी। स्टार्टअप्स, लघु उद्यमों और विनिर्माण को समर्थन देने के लिए प्रोत्साहन और फंड की व्यवस्था की जाएगी। राष्ट्रीय विनिर्माण मिशन की स्थापना की जाएगी। इन्फ्रास्ट्रक्चर और ऊर्जा: इन्फ्रास्ट्रक्चर विकास और क्षेत्रीय हवाई संपर्क को बढ़ावा दिया जाएगा। महत्वपूर्ण खनिजों के विकास पर ध्यान दिया जाएगा। न्यूक्लियर एनर्जी मिशन के तहत 2047 तक 100 गीगावाट परमाणु ऊर्जा उत्पादन का लक्ष्य रखा गया है। आर्थिक अनुमान: सकल घरेलू उत्पाद (GDP) में 10.1% की नाममात्र वृद्धि का अनुमान है। वित्तीय घाटा GDP का 4.4% रहने की उम्मीद है। कुल बजट व्यय 50.65 ट्रिलियन रुपये होने का अनुमान है। इस बजट के माध्यम से सरकार का उद्देश्य आर्थिक विकास को गति देना, मध्यम वर्ग को राहत प्रदान करना और विभिन्न क्षेत्रों में सुधार लाना है।
"""

# text = """नया कर स्लैब:
# 1.2 मिलियन रुपये तक की वार्षिक आय पर कोई कर नहीं।
# 2.4 मिलियन रुपये से अधिक की आय पर अधिकतम 30% कर दर लागू होगी।"""

font_path = "/usr/share/fonts/truetype/samyak/Samyak-Devanagari.ttf"

def gen_audio(audio_path):
    tts = gTTS(text, lang="hi")  # Change 'hi' for Hindi, 'en' for English
    tts.save(audio_path)
    audio = AudioFileClip(audio_path)
    return audio

def gen_subtitle(duration):
    total_duration = duration
    words = text.split()
    num_words = len(words)
    words_per_second = len(words) / total_duration

    subtitles = []
    start_time = 0
    chunk_size = 4  # Number of words per subtitle

    for i in range(0, len(words), chunk_size):
        subtitle_text = " ".join(words[i:i+chunk_size])
        end_time = start_time + (chunk_size / words_per_second)

        subtitle_clip = TextClip(text=subtitle_text, font=font_path, font_size=40, color='orange',method="caption", bg_color="black", size=(1280, 100))
        subtitle_clip = (
            subtitle_clip.with_position(("center", "bottom"))
            .with_start(start_time)
            .with_duration(end_time - start_time)  # Ensure proper timing
        )

        subtitles.append(subtitle_clip)
        start_time = end_time  # Move to next subtitle chunk
    return subtitles

def gen_video(audio, subtitles, image_files):
    total_duration = audio.duration
    #image_durations = np.linspace(0, audio.duration, len(image_files) + 1)[1:]
    image_duration = total_duration / len(image_files)
    image_clips = [ImageClip(img).with_duration(image_duration) for img in image_files]

    # 5️⃣ Concatenate Images to Make Video
    video = concatenate_videoclips(image_clips, method="compose").with_audio(audio)

    # 6️⃣ Merge Video & Subtitles
    final_video = CompositeVideoClip([video] + subtitles)
    final_video.duration = total_duration
    # 7️⃣ Export Final Video
    final_video.write_videofile(f"{image_directory}/faceless_video.mp4", fps=24, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    image_directory = "/home/gmohit/Pictures/images"
    image_files = [f"{image_directory}/1.jpg",
                   f"{image_directory}/2.jpg",
                   f"{image_directory}/3.jpg",
                   f"{image_directory}/4.jpg",
                   f"{image_directory}/5.jpg",
                   f"{image_directory}/6.jpg",
                   f"{image_directory}/7.jpg"
                   ]
    audio_path = "output_audio.mp3"
    audio = gen_audio(audio_path)
    subtitles = gen_subtitle(audio.duration)
    gen_video(audio, subtitles, image_files)