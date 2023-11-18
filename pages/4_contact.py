import streamlit as st
st.set_page_config(page_title="RetinaVision AI - Contact", page_icon="icons/braille-solid.svg", initial_sidebar_state='collapsed')

css_title = '''
<style>
.footer-title {
    font-size: 30px;
    color: #888888;
    text-align: center;
}
</style>
'''
st.markdown(css_title, unsafe_allow_html=True)
st.markdown('<H1 class="footer-title"><i class="fa-solid fa-envelope"></i>&nbsp&nbspGet in Touch!</H1>', unsafe_allow_html=True)
st.markdown('')

contact_form = """
<form action="https://formsubmit.co/shreyasdb99@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="text" name="name" placeholder="Your name" required>
    <input type="email" name="email" placeholder="Your email" required>
    <textarea name="message" placeholder="Your message here"></textarea>
    <button type="submit">Send</button>
</form>
"""

st.markdown(contact_form, unsafe_allow_html=True)

# Use local CSS file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('***')

css_copyr = '''
<style>
.footer-text {
    font-size: 15px;
    color: #888888;
    text-align: center;
}
</style>
'''

st.markdown(css_copyr, unsafe_allow_html=True)

st.markdown('<p class="footer-text">Copyright © 2023 &nbsp<i class="fa-solid fa-braille"></i>&nbspRetinaVisionAI</p>', unsafe_allow_html=True)
st.markdown("<p class='footer-text'>Contact us at shreyasdb99@gmail.com</p>", unsafe_allow_html=True)
st.markdown('')


css_fa = '''                                                                                                                                                     
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
.footer-fa {
    font-size: 20px;
    color: #888888;
    text-align: center;
    margin: 0 5px;
    display: inline-block;
}
.footer-icons {
    text-align: center;
}
</style>
<div class="footer-icons">                                                                                                                                                                                                                                                                                               
    <a href="https://github.com/shre-db" target="_blank"><i class="fa-brands fa-github footer-fa"></i></a>                                                                                                                                                                
    <a href="https://www.linkedin.com/in/shreyas-bangera-aa8012271/" target="_blank"><i class="fa-brands fa-linkedin footer-fa"></i></a>                                                                                                                                                                         
    <a href="https://www.instagram.com/shryzium/" target="_blank"><i class="fa-brands fa-instagram footer-fa"></i></a>
</div><br>
<div>
    <p class="footer-text">Version 1.1.0</p>
</div>
'''

st.markdown(css_fa, unsafe_allow_html=True)