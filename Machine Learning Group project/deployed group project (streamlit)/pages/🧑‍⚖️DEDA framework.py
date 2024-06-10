# Done by Amy

import streamlit as st


st.title ( ":green[DEDA framework Legal and ethics]")


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Intro","Algorithms, Sources and Anonymization", 
"Visualization","Sharing, Reusing And Re Purposing"
,"Responsibility","Transparency and Privacy"
,"Bias","Conclusion"])


with tab1:
    st.markdown("#### STEP 1: ####")
    st.write ("Select one person within your group who will write down the answers - :violet[Amy with the help of my team members - Hubert, Matey, Raphael and Maikel.]")
    st.markdown ("-----------")
    st.markdown ("#### STEP 2: ####")

    st.markdown ("**:red[Q1. Project name, date, place?]**")

    st.markdown (" ##### ` Project name` - The Crime Crystal Ball. #####")
    st.markdown ("##### `Place` - Breda University of Applied Sciences, Breda. ##### ")
    st.markdown ("-----------")
    st.markdown ("**:red[Q2. Participants of the project?]** ")
    st.markdown ("##### **:violet[The Panic Room - Hubert Waleńczak, Matey Nedyalkov, Raphaël van Rijn , Maikel Boezer, Amy Suneeth.]**")
    st.markdown ("**:red[Q3. What is the project about and what is its goal?]**")
    st.markdown ("The primary objective of the project is to assist the municipality of" 
    "Breda in improving the standard of quality of life for its residents." 
    "Our team is focusing on the crime rates of Breda and hence creating a prediction model for predicting crimes." 
    "This model will involve some factors such as unemployment rates, education and income"
    "from different neighborhoods of Breda.")
    st.markdown("**:red[Q4. What kind of data will you be using?]**")
    st.markdown('We will be using different data such as the crime rates in Breda per year, unemployment'
    'rates, education and income.')
    st.markdown("**:red[Q5. Who might be affected by the project?]**")
    st.markdown("People with low income, the boa's, people with low to no education and unemployed people")
    st.markdown("**:red[Q6. What are the benefits of the project? What are the problems or concerns that might arise"
    "in connection with this project?]**")
    st.markdown('The project is intended to help the municipality to increase the quality of life'
    'for its residents. The main problems that we might have to deal with will be the lack of required data to create a prediction model.')
    st.markdown("-----------")


with tab2: 
    st.subheader("*:green[The algorihm considerations when it comes to the DEDA ethical framework]*")

    st.markdown('**:blue[Q1. Does the project make use of an algorithm, or some form of machine learning?]**')

    st.markdown("Yes, a clustering and prediction model")
    st.markdown("**:blue[Q2. Is there someone within the team that can explain how the algorithm or neural networks? If not, go to ‘Source.’ in question works?]**")
    st.markdown("Yes")
    st.markdown("**:blue[Q3. Is there someone who can provide an explanation that is accessible to the wider public?]**")
    st.markdown("Yes")
    st.subheader("*:green[The sources we considered while adhering to the DEDA ethical framework]*")
    st.markdown("**:blue[Q1. Where do the data(sets) come from?]**")
    st.markdown("The dataset comes from the Gemeente website and polite.nl ( CBS database).")
    st.markdown("**:blue[Q2. Do the data have an expiration date?]**")
    st.markdown("No")

    st.subheader("*:green[Ethical Anonymization with regards to the DEDA framework]*")
    st.markdown("**:blue[Q1. Should the data be anonymized, pseudonymized or generalized?]**")
    st.markdown("No")
    st.markdown("**:blue[Q2. Who has access to the encryption key to de-pseudonymize the data?]**")
    st.markdown("It is open source data hence the question is not accessible for this particular project")


with tab3:
    st.subheader("*:green[The things taken into consideration with regard to the DEDA framewrok while visualizing data]*")
    st.markdown("**:violet[Q1. How will the results of the project be presented? Are the results suitable for visualization?]**")
    st.markdown("The final project will consist of a Dashboard which shows different information regarding factors influencing crime in different neighborhoods. And the final products are also suitable for visualization.")
    st.markdown("**:violet[Q2. What alternative ways of visualizing the results are there?]**")
    st.markdown("Map of Breda with their classifications and factors.")
   
with tab4:
    st.subheader("*:green[The stratergies taken up to ensure ethical Sharing, Reusing and Re Purposing ]*")
    st.markdown("**:red[Q1. Are any of the data suitable for reuse? If so, under what conditions and for what (new) purpose(s) could they be reused?]**")
    st.markdown("All the data that we are using are reusable and can be used to make different models of prediction using the same.")
    st.markdown("**:red[Q18. Are there any obligations (not) to make the data publicly available? If you were to provide open access to parts of the data, what opportunities and risks might arise?]**")
    st.markdown("Since the data is open and accessible to everyone there is no obligation not to make the data publicly unavailable. There won't be any risks involved when it comes to providing open access to the data that we are using since it is already open and available for everyone to see.")
    

with tab5:
    st.subheader("*:green[The different Crucial Responsibilities taken up for Establishing a Lawful Model.]* ")
    st.markdown("**:blue[Q1. Which laws and regulations apply to your project?]**")
    st.markdown("For this project we are implementing the GDPR ethical guidelines. GDPR stands for General Data Protection Regulation. It is a comprehensive data protection law that was implemented in the European Union. The GDPR aims to safeguard the privacy and personal data of individuals within the EU and regulates how organizations collect, process, and handle such data.")
    st.markdown("**:blue[Q2. Are the duties and responsibilities of that person clear, with regard to this project?]**")
    st.markdown("Yes. We have made sure to ensure that we follow the GDPR guidelines. We have also tried to make our work transparent and understandable for everyone.")
    st.markdown("**:blue[Who is ultimately responsible for the project?]**")
    st.markdown("The project manager for our group is Raphael. However, everyone within our group is equally responsible for everything regarding the project.")
    st.markdown("**:blue[Q3. Are the duties and responsibilities of that person clear, with regard to this project.]**")
    st.markdown("Yes. We have made sure to ensure that we follow the GDPR guidelines. We have also tried to make our work transparent and understandable for everyone to see.")


with tab6:
    st.subheader("*:green[The different stratergies taken up by the team to ensure transparency when it comes to our machine learnign model.]* ")
    st.markdown("**:violet[Q1. Does the project risk generating public concern or outrage?]** ")
    st.markdown("No. Only for criminals :)")
    st.markdown("**:violet[Q2. How transparent are you about this project towards citizens?]**")
    st.markdown('All the data is open source and everyone will know what factors will be embedded in our unsupervised machine learning model. So, yes, the project is very transparent towards citizens.')
    st.markdown('**:violet[Q3. Do citizens have the opportunity to raise objections to the results of the project?]**')
    st.markdown('As our project operates in collaboration with the municipality, they serve as a primary point of contact for inquiries and feedback. This professional channel ensures a transparent and accountable process for addressing any issues related to our project.')
    st.markdown('**:violet[Q4. Can citizens opt out of their involvement in the project? If so, when and how can they do this?]**')
    st.markdown('This is not possible since we have not collected any kind of personal information and our data is mostly on a neighborhood level. So it would very unlikely for any one person/citizen to have objections')
    
    st.subheader('*:green[The stratergies we implementes to ensure privacy of the citizens of Breda.]*')
    st.markdown('**:violet[Q1. Is there a data protection officer or data privacy officer involved in this project.]**')
    st.markdown("We do not have any data protection officer. However we actively ensure to follow the guidelines.")
    st.markdown('**:violet[Q2. Have you conducted a PIA (Privacy Impact Assessment) or DPIA (Data Protection Impact Assessment)?]**')
    st.markdown('Since we are not using any personal information for this project, it is not necessary for us to conduct these guidelines. We however ensure to follow the ethical guidelines mentioned in the GDPR framework.')
    st.markdown('**:violet[Q3. Does this project make use of personal data? If not, continue with “Bias.]**')
    st.markdown('No')



with tab7:
    st.subheader("*:green[The methods we followed to ensure that there is no bias in the dataset used for this project.]*")
    st.markdown("**:red[Q1 As a member of the project, what outcomes do you expect?]**")
    st.markdown("I expect us to have a machine learning model and a dashboard containing all the information that we used along with analysis for the same.")
    st.markdown("**:red[Q2. Is there anything about this project that makes you uneasy?]** ")
    st.markdown("No")
    st.markdown("**:red[Q3. Will the results of the analysis be evaluated by a human before being implemented?]**")
    st.markdown("Yes - The municipality.")
    st.markdown("**:red[Q4. Is there a risk that your project could contribute to discrimination against certain people or groups?]** ")
    st.markdown("No, because the data is on a neighborhood basis.")
    st.markdown("**:red[Q5. Are all relevant citizens adequately represented within your data(sets)? Which ones are missing or under-represented?]**")
    st.markdown("There are no underrepresented citizens in regards to our project.")
    st.markdown("**:red[Q6. Is there a feedback loop in the model that might have negative consequences?]** ")
    st.markdown("No")
    st.markdown("**:red[Q7. Are you gathering the information that is appropriate for the purpose of your project?]** ")
    st.markdown("Yes")
    st.markdown("**:red[Q8. Is there a risk that the project will unintentionally create incentives for undesirable behavior?]** ")
    st.markdown("Yes.This may happen if we are able to find out the areas with higher crimes and if the municipality is able to do something about it, the people who get caught might have incentives for undesirable behavior.")
    st.markdown("**:red[Q9. Function creep: can you imagine a future scenario in which the results of your project could be (mis)used for alternative purposes?]**")
    st.markdown("No")
    st.markdown("**:red[Q10 Do your answers change when you consider possible long-term effects? Why?]**")
    st.markdown("No")


with tab8:
    st.markdown("## STEP : 3 ##")
    st.markdown("## *:green[Which values and principles are important to you personally, and which are important to your organization?]*")
    








