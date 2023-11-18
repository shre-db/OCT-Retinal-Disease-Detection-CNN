import streamlit as st
st.set_page_config(page_title="RetinaVision AI - Privacy", page_icon="icons/braille-solid.svg", initial_sidebar_state='collapsed')


st.title("Privacy Notice")
st.markdown('')
st.markdown("""
    **Image Data Storage and Deletion**

    At RetinaVision AI, we deeply value your privacy and want to ensure transparency regarding the handling
    of your image data. This Privacy Notice explains how we collect, use, and retain your image data and our commitment
    to safeguarding your information.       

    **Image Data Collection**

    When you use our service, you may upload images for purposes such as image classification. These images are temporarily 
    stored on our servers solely for the purpose of performing the tasks
    you request.

    **Data Retention**

    We respect your privacy and are committed to retaining your image data only for the necessary duration required
    to complete the requested tasks. Your image data will be automatically and securely deleted from our systems when you exit the session.

    **Data Security**

    We prioritize the security of your image data. Our web app is hosted on Streamlit Community Cloud, which provides
    industry-standard security measures to protect against unauthorized access and data breaches. Your data is treated with
    the utmost care to ensure its confidentiality and integrity while hosted on Streamlit Community Cloud.
    
    Streamlit Community Cloud is a trusted platform that maintains high standards of security and data protection. 
    However, please be aware that when using our service, your data may be subject to Streamlit Community Cloud's privacy and
    security policies. We recommend reviewing Streamlit Community Cloud's terms of service and privacy policy for more information.
    
    Rest assured that we are committed to providing a secure and reliable environment for your data while using Streamlit Community Cloud for hosting.
    If you have any questions or concerns about the security of your data, please don't hesitate to contact us. Your privacy and data security are of utmost importance to us.


    **No Sharing or Third-Party Access**

    We do not share, sell, or provide access to your image data to any third parties, except when required by law or
    with your explicit consent. Your data remains strictly confidential and is used solely for the purposes you intend
    when using our service.

    **User Control**

    You have the right to control your data. If you have any concerns about the storage or deletion of your image data,
    please don't hesitate to contact us. We are here to address your questions and requests.

    **Cookie Usage**

    We want to assure you that our website does not use cookies or similar tracking technologies. Your browsing experience
    on our website is cookie-free. You can enjoy our services without the need to manage or modify any cookie preferences through your browser settings.


    **Changes to this Privacy Notice**

    We may update this Privacy Notice from time to time to reflect changes in our practices and services. Any changes
    will be communicated through our website or other appropriate means.

    By using our service, you consent to the terms outlined in this Privacy Notice. If you do not agree with any part
    of this notice, please refrain from using our services.

    If you have any questions or concerns about your privacy or data management, please reach out to us
    at shreyasdb99@gmail.com.

    Thank you for choosing RetinaVision AI. We are dedicated to safeguarding your privacy and ensuring a
    secure and enjoyable experience with our services.
    
    Sincerely,
""")


css_signature = '''
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<style>
.footer-signature {
    font-size: 17px;
    color: #000000;
    text-align: left;
}
</style>
<p class="footer-signature"><i class="fa-solid fa-braille"></i>&nbspRetinaVision AI</p>
'''
st.markdown(css_signature, unsafe_allow_html=True)


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


st.markdown('<p class="footer-text">Copyright © 2023 &nbsp<i class="fa-solid fa-braille"></i>&nbspRetinaVision AI</p>', unsafe_allow_html=True)
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