st.sidebar.radio('', ['Walking','Running', 'Cycling'])
st.sidebar.radio('Are you...', ['Local',' Tourist (we optimise your route to help you discover places popular with tourists)'])


# Moar items for the sidebar

if st.sidebar.checkbox('Amenities'):
    st.sidebar.slider ( "Public toilets" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Water fountain" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
if st.sidebar.checkbox('Food'):
    st.sidebar.slider ( "Bubble tea " , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Hawker centres" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Cafes" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Bars and pubs" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
if st.sidebar.checkbox('Attractions'):
    st.sidebar.slider ( "Shopping centers " , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Museums" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Parks" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )
    st.sidebar.slider ( "Events" , min_value=1 , max_value=10 , value=10 , step=1 , format=None , key=None )