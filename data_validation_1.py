import logging
import random
import pandas as pd
import streamlit as st
from googletrans import Translator
from langdetect import detect
# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import io

# Set wide layout as the first Streamlit command
st.set_page_config(layout="wide")


def main():
    # Sidebar Navigation
    navigation = st.sidebar.radio("Go to", ["Home", "Data Profile", "Column-Wise Analysis", "Checklist", "Data Validated"])

    # Initialize session state for storing home page data
    if 'home_data' not in st.session_state:
        st.session_state['home_data'] = {}

    # Initialize session state for column-wise analysis data
    if 'column_wise_analysis_data' not in st.session_state:
        st.session_state['column_wise_analysis_data'] = None

    # Initialize session state for checklist data
    if 'checklist_answers' not in st.session_state:
        st.session_state['checklist_answers'] = {}

    if navigation == "Home":
        st.title("Welcome to the Data Validation App")
        st.write("Use the sidebar to navigate between sections.")

        # Input form on the Home page
        with st.form("home_form"):
            link_to_data = st.text_input('Link to Data')
            client_name = st.text_input('Client Name & Industry')
            event_type = st.text_input('Type of Event Described in Data')
            num_records = st.number_input('Total Number of Records', min_value=0, value=0)
            date_received = st.date_input('Date of Receiving')
            app_url = st.text_input('App URL')
            project_owner = st.text_input('Project Owner')
            senior_analyst = st.text_input('Senior Analyst')
            validated_by = st.text_input('Sheet Validated by')

            # Save button
            submitted = st.form_submit_button("Save")
            if submitted:
                # Save the values in session state
                st.session_state['home_data'] = {
                    'Link to Data': link_to_data,
                    'Client Name & Industry': client_name,
                    'Type of Event Described in Data': event_type,
                    'Total Number of Records': num_records,
                    'Date of Receiving': date_received,
                    'App URL': app_url,
                    'Project Owner': project_owner,
                    'Senior Analyst': senior_analyst,
                    'Sheet Validated by': validated_by
                }
                st.success('Data saved successfully!')

    elif navigation == "Data Profile":
        data_profile()

    elif navigation == "Column-Wise Analysis":
        column_wise_analysis()

    elif navigation == "Checklist":
        checklist(st.session_state.get('column_wise_analysis_data'))

    elif navigation == "Data Validated":
        st.title("Data Validated")

        # Display Home Page Data
        st.subheader("Home Page Data")
        if 'home_data' in st.session_state and st.session_state['home_data']:
            home_data = st.session_state['home_data']
            home_data_df = pd.DataFrame(list(home_data.items()), columns=['Field', 'Value'])
            st.dataframe(home_data_df)
        else:
            st.warning("No Home Page data available.")

        # Display Column-Wise Analysis Data
        st.subheader("Column-Wise Analysis Data")
        if 'column_wise_analysis_data' in st.session_state and st.session_state['column_wise_analysis_data'] is not None:
            st.dataframe(st.session_state['column_wise_analysis_data'])
        else:
            st.warning("No Column-Wise Analysis data available.")

        # Display Checklist Data
        st.subheader("Checklist Data")
        if 'checklist_answers' in st.session_state and st.session_state['checklist_answers']:
            checklist_answers = st.session_state['checklist_answers']
            checklist_df = pd.DataFrame.from_dict(checklist_answers, orient='index', columns=['Answer', 'Extra Details'])
            checklist_df.index.name = 'Question'
            checklist_df.reset_index(inplace=True)
            st.dataframe(checklist_df)
            # st.dataframe(st.session_state[checklist_df])
        else:
            st.warning("No Checklist data available.")

        if st.button("Prepare CSV"):
            if 'home_data' in st.session_state and 'checklist_answers' in st.session_state:
                # Prepare DataFrames
                home_df = pd.DataFrame.from_dict(st.session_state['home_data'], orient='index', columns=['Value'])
                home_df.index.name = 'Field'
                home_df.reset_index(inplace=True)

                checklist_df = pd.DataFrame.from_dict(st.session_state['checklist_answers'], orient='index')
                checklist_df.index.name = 'Question'
                checklist_df.reset_index(inplace=True)

                # Initialize a CSV buffer
                buffer = io.StringIO()

                # Add Home Page Data to CSV
                home_df.to_csv(buffer, index=False, header=True)
                buffer.write("\n\n")  # Add a separator between sections

                # Add Checklist Data to CSV
                checklist_df.to_csv(buffer, index=False, header=True)
                buffer.write("\n\n")  # Add a separator between sections

                # Add Column-Wise Analysis Data to CSV
                if st.session_state['column_wise_analysis_data'] is not None:
                    column_wise_data = st.session_state['column_wise_analysis_data']
                    column_wise_data.to_csv(buffer, index=False, header=True)
                else:
                    buffer.write("No Column-Wise Analysis Data available.\n")

                buffer.seek(0)
                st.download_button(
                    label="Download All Data as CSV",
                    data=buffer.getvalue(),
                    file_name="data_validated.csv",
                    mime="text/csv"
                )


# Translator
translator = Translator()

def translate_to_english_batch(text_list):
    if not text_list or all(pd.isna(t) for t in text_list):
        return text_list

    try:
        valid_texts = [t for t in text_list if isinstance(t, str) and t.strip()]
        if not valid_texts:
            return text_list

        translations = translator.translate(valid_texts, src='auto', dest='en')
        translation_dict = dict(zip(valid_texts, [t.text for t in translations]))

        return [translation_dict.get(t, t) if isinstance(t, str) and t.strip() else t for t in text_list]
    except Exception as e:
        logging.error(f"Batch translation error: {e}")
        return text_list

def translate_to_english(text):
    if not isinstance(text, str) or not text.strip():
        return text

    try:
        lang = detect(text)
        if lang != 'en':
            translation = translator.translate(text, src=lang, dest='en')
            return translation.text
        return text
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

def data_profile():
    global agg_data
    st.title("Data Profile")

    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], key="file_uploader_1")

    if uploaded_file is not None:
        st.subheader("Imported File")

        # Determine file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
                    return
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                return
        elif uploaded_file.name.endswith('.xlsx'):
            try:
                df = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
                return
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return

        # Store the DataFrame in session state
        st.session_state.uploaded_df = df
        st.write(df)

        # Convert all object columns to strings
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)

        # st.dataframe(df)

        st.subheader("DataFrame Info")

        st.write("Number of records:", df.shape[0])
        st.write("Number of columns:", df.shape[1])

        random_examples = {}
        for column in df.columns:
            if df[column].notnull().any():
                random_example = random.choice(df[column].dropna().values.tolist())
                random_examples[column] = random_example
            else:
                random_examples[column] = None

        data = {
            'Data types (pandas Identified)': df.dtypes,
            'Number of values': df.count(),
            'Number of unique values': df.nunique(),
            'Dataset Example': random_examples
        }
        # Plotting the unique values for each column
        # st.subhead("Unique Values per Column")

        unique_values = df.nunique()

        # Create a dropdown for column selection
        # st.subhead("Select Columns to Generate a Graph")

        # Multi-select widget for selecting columns
        selected_columns = st.multiselect(
            "Select one or more columns to plot",
            options=df.columns.tolist(),
            default=df.columns.tolist(),  # Set default to all columns
            help="You can select multiple columns to plot"
        )

        # Layout for selecting data type and chart type
        col1, col2 = st.columns([2, 1])  # Adjust the column widths as needed

        with col1:
            graph_type = st.selectbox(
                "Select the type of data to plot",
                options=["Unique Values", "Total Population", "Both"]
            )

        with col2:
            chart_type = st.selectbox(
                "Select the type of chart",
                options=["Bar", "Pie"]
            )

        # Initialize empty lists to store values for plotting
        if selected_columns:
            selected_data = df[selected_columns]

            # Calculate unique values and total population
            unique_counts = selected_data.nunique()
            total_counts = selected_data.count()

            # Initialize the figure
            fig = None

            if graph_type in ["Unique Values", "Both"]:
                if chart_type == "Bar":
                    fig_unique = px.bar(
                        x=unique_counts.index,
                        y=unique_counts.values,
                        labels={'x': 'Column Name', 'y': 'Count'},
                        title='Number of Unique Values per Selected Column(s)'
                    )
                    fig_unique.update_layout(
                        width=1000,
                        height=600,
                        xaxis_title="Column Name",
                        yaxis_title="Unique Values Count"
                    )
                    fig = fig_unique
                elif chart_type == "Pie":
                    fig_unique = px.pie(
                        names=unique_counts.index,
                        values=unique_counts.values,
                        title='Number of Unique Values per Selected Column(s)'
                    )
                    fig_unique.update_layout(
                        width=1000,
                        height=600
                    )
                    fig = fig_unique

            if graph_type in ["Total Population", "Both"]:
                if fig is None:
                    if chart_type == "Bar":
                        fig_total = px.bar(
                            x=total_counts.index,
                            y=total_counts.values,
                            labels={'x': 'Column Name', 'y': 'Count'},
                            title='Total Population per Selected Column(s)'
                        )
                        fig_total.update_layout(
                            width=1000,
                            height=600,
                            xaxis_title="Column Name",
                            yaxis_title="Total Population Count"
                        )
                        fig = fig_total
                    elif chart_type == "Pie":
                        fig_total = px.pie(
                            names=total_counts.index,
                            values=total_counts.values,
                            title='Total Population per Selected Column(s)'
                        )
                        fig_total.update_layout(
                            width=1000,
                            height=600
                        )
                        fig = fig_total
                else:
                    if chart_type == "Bar":
                        fig.add_bar(
                            x=total_counts.index,
                            y=total_counts.values,
                            name='Total Population',
                            marker_color='rgba(55, 83, 109, 0.5)'
                        )
                        fig.update_layout(
                            barmode='group',
                            xaxis_title="Column Name",
                            yaxis_title="Count"
                        )
                    elif chart_type == "Pie":
                        fig_total = px.pie(
                            names=total_counts.index,
                            values=total_counts.values,
                            title='Unique Values and Total Population per Selected Column(s)'
                        )
                        fig_total.update_layout(
                            width=1000,
                            height=600
                        )
                        fig = fig_total

            if fig:
                st.plotly_chart(fig)

            # Custom graph
            st.subheader("Custom Graph Creation")

            # Create two columns for X axis and Y axis dropdowns
            col1, col2 = st.columns(2)

            # Dropdown for selecting X Axis in the first column
            with col1:
                x_axis = st.selectbox(
                    "Select X Axis",
                    options=df.columns.tolist(),
                    help="Select the column for the X axis"
                )

            # Dropdown for selecting Y Axis in the second column
            with col2:
                y_axis: object = st.selectbox(
                    "Select Y Axis",
                    options=df.columns.tolist(),
                    help="Select the column for the Y axis"
                )

            # Create a dropdown for selecting the aggregation type
            aggregation_type = st.selectbox(
                "Select Aggregation Type",
                options=["Count", "Sum", "Unique Values"],
                help="Select how the data should be aggregated"
            )

            if not st.button("Generate Graph"):
                pass
            # Generating the graph
            else:
                if aggregation_type == "Count":
                    agg_data = df.groupby(x_axis)[y_axis].count()
                elif aggregation_type == "Sum":
                    agg_data = df.groupby(x_axis)[y_axis].sum()
                elif aggregation_type == "Unique Values":
                    agg_data = df.groupby(x_axis)[y_axis].nunique()

                # Plotting the data with multiple colors for each bar
                st.bar_chart(agg_data, use_container_width=True)

            # Create the graph based on selections
            if x_axis and y_axis and aggregation_type:
                if aggregation_type == "Count":
                    data_to_plot = df.groupby(x_axis).size().reset_index(name='Count')
                    fig = px.bar(
                        data_to_plot,
                        x=x_axis,
                        y='Count',
                        labels={x_axis: x_axis, 'Count': 'Count'},
                        title=f'Count of {y_axis} by {x_axis}'
                    )
                elif aggregation_type == "Sum":
                    data_to_plot = df.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.bar(
                        data_to_plot,
                        x=x_axis,
                        y=y_axis,
                        labels={x_axis: x_axis, y_axis: f'Sum of {y_axis}'},
                        title=f'Sum of {y_axis} by {x_axis}'
                    )
                elif aggregation_type == "Unique Values":
                    data_to_plot = df.groupby(x_axis)[y_axis].nunique().reset_index(name='Unique Count')
                    fig = px.bar(
                        data_to_plot,
                        x=x_axis,
                        y='Unique Count',
                        labels={x_axis: x_axis, 'Unique Count': 'Unique Values Count'},
                        title=f'Unique Values of {y_axis} by {x_axis}'
                    )

                st.plotly_chart(fig)

        df_info = pd.DataFrame.from_dict(data, orient='index')
        st.write(df_info)

        if st.button('Start Data Cleaning'):
            with st.spinner('Cleaning data...'):
                cleaned_df, progress_data = clean_data(df)
                st.session_state.cleaned_df = cleaned_df

                st.subheader("Progress Graph")
                st.line_chart(progress_data.set_index('Step'))

                st.subheader("Cleaned Data")
                st.dataframe(cleaned_df)

                csv = cleaned_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download cleaned file as CSV",
                    data=csv,
                    file_name='cleaned_data_profile.csv',
                    mime='text/csv',
                    key="download_button_cleaned"
                )
        else:
            st.write("You can proceed directly to the Column-Wise Analysis.")

def clean_data(df):
    logging.info("Starting data cleaning...")

    num_rows = df.shape[0]
    progress_data = []
    num_columns = len(df.columns)

    progress_bar = st.progress(0)

    for idx, column in enumerate(df.columns):
        if df[column].dtype == object:
            text_entries = df[column].dropna().tolist()
            translated_texts = translate_to_english_batch(text_entries)
            df[column].update(pd.Series(translated_texts, index=df[column].dropna().index))

        progress = (idx + 1) / num_columns
        progress_bar.progress(min(max(progress, 0), 1))
        progress_bar.progress(progress)
        progress_data.append({'Step': f'Processing {column}', 'Progress': progress})

    date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    other_columns = [col for col in df.columns if col not in date_columns]

    for idx, column in enumerate(date_columns):
        df[column] = pd.to_datetime(df[column], errors='coerce').dt.strftime('%m/%d/%Y')
        progress = (idx + 1) / len(date_columns) * 100
        progress = min(max(progress, 0), 100)
        progress_data.append({'Step': f'Converting {column}', 'Progress': progress})

    for column in other_columns:
        pass

    progress_bar.progress(100)
    progress_data.append({'Step': 'Completed', 'Progress': 100})

    progress_df = pd.DataFrame(progress_data)
    logging.info("Data cleaning completed.")

    return df, progress_df


# Global variable to track example indices for each column
example_index = {}


def column_wise_analysis():
    st.title("Column-Wise Analysis")

    if 'cleaned_df' in st.session_state:
        df = st.session_state.cleaned_df
    elif 'uploaded_file' in st.session_state:
        df = st.session_state.uploaded_file
    else:
        st.write("Please upload and process a CSV file in the Data Profile section first.")
        return

    # Initialize example_index dictionary
    global example_index
    example_index = {}

    # User input section for column-wise analysis
    input_values = {}

    for column in df.columns:
        col_container = st.columns(3)  # Adjusted to show three columns in a row

        with col_container[0]:
            # Display the column name
            st.markdown(f"<p style='font-size: larger; font-weight: bold;'>{column}</p>", unsafe_allow_html=True)

            # Dataset Example section below the column name
            example_index[column] = random.randint(0, len(df[column]) - 1)  # Get a random index for the example
            st.markdown("**Dataset Example**")
            example_value = df[column].iloc[example_index[column]]
            st.write(f": {column}: {example_value}")

        with col_container[1]:
            input_value1 = st.text_input(f"Enter the description for {column}", value=column, key=f"{column}_1")

            data_types = ["", "Integer", "Float", "String", "Categorical", "Alphanumeric", "Date", "Timestamp", "Other"]
            inferred_type = infer_data_type(df[column])
            input_value2 = st.selectbox(f"Select the data type for {column}", options=data_types,
                                        index=data_types.index(inferred_type), key=f"{column}_2")

            input_value3 = st.text_input(f"What can be mined from {column}", key=f"{column}_3")

            # Additional fields for analysis
            input_value4 = st.checkbox(f"Potential Bangers (Yes/No) for {column}", key=f"{column}_4")
            input_value5 = st.text_input(f"Enter Banger Titles for {column}", key=f"{column}_5")

        with col_container[2]:
            total_values = df[column].count()
            unique_values = df[column].nunique()
            st.write(f"Total Values: {total_values}")
            st.write(f"Unique Values: {unique_values}")

            # Expanders for delayed input display
            with st.expander(f"How the field is made for {column}"):
                input_value6 = st.text_area(f"How the field is made for {column}", key=f"{column}_6")

            with st.expander(f"Comments for {column}"):
                input_value7 = st.text_area(f"Comments for {column}", key=f"{column}_7")

            with st.expander(f"Questions for client for {column}"):
                input_value8 = st.text_area(f"Questions for client for {column}", key=f"{column}_8")

        st.markdown("***")

        # Store all input values in the dictionary
        input_values[column] = [input_value1, input_value2, input_value3, input_value4, input_value5, total_values,
                                unique_values, input_value6, example_value, input_value7, input_value8]

    # Convert input values to DataFrame
    input_df = pd.DataFrame(input_values, index=['Description', 'Data Type', 'What can be mined', 'Potential Bangers',
                                                 'Banger Titles', 'Total Values', 'Unique Values',
                                                 'How the field is made', 'Dataset Example',
                                                 'Comments', 'Questions for client']).T

    # Store the DataFrame in session_state so it can be used in the Checklist section
    st.session_state.column_wise_analysis_data = input_df

    # Display the DataFrame for user edits
    editable_df = st.data_editor(input_df)

    # Add a Save button
    if st.button("Save"):
        st.session_state.column_wise_analysis_data = editable_df
        st.success("Data saved successfully!")

    # Download button for CSV
    csv = editable_df.to_csv().encode('utf-8')
    st.download_button(
        label="Download file as CSV",
        data=csv,
        file_name='column_analysis.csv',
        mime='text/csv',
    )


def infer_data_type(series):
    if pd.api.types.is_integer_dtype(series):
        return 'Integer'
    elif pd.api.types.is_float_dtype(series):
        return 'Float'
    elif pd.api.types.is_string_dtype(series):
        return 'String'
    elif pd.api.types.is_categorical_dtype(series):
        return 'Categorical'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'Timestamp'
    elif pd.api.types.is_object_dtype(series):
        return 'Alphanumeric'
    else:
        return 'Other'


def update_example(example_container, series, index, column):
    # Initialize example_index if not already done
    if column not in example_index:
        example_index[column] = 0

    # Show random data example from the column
    examples = series.dropna().astype(str).unique()
    if len(examples) > 0:
        example_container.write(examples[index])
    else:
        example_container.write("No data available.")

    # Add navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button('<', key=f'{column}_prev'):
            index = (index - 1) % len(examples)
            example_index[column] = index
            update_example(example_container, series, index, column)
    with col2:
        if st.button('>', key=f'{column}_next'):
            index = (index + 1) % len(examples)
            example_index[column] = index
            update_example(example_container, series, index, column)


def checklist(input_df=None):
    st.title("Checklist")

    # CSS to make the Column-Wise Analysis fixed at the top
    st.markdown("""
        <style>
        .fixed-header {
            position: sticky;
            top: 0;
            background-color: white;
            z-index: 100;
            padding: 10px;
            border-bottom: 1px solid #f0f0f0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Container for the fixed Column-Wise Analysis Data
    with st.container():
        st.subheader("Column-Wise Analysis Data")
        if input_df is not None:
            st.markdown('<div class="fixed-header">', unsafe_allow_html=True)
            st.dataframe(input_df)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No data available to display")

    # Define the checklist questions with custom dropdown options for each section
    checklist_data = [
        [1, "Is primary key present?", ["Yes", "No"], ""],
        [2, "Are there interesting columns describing the product / model or the event?",
         ["Yes, there is at least 1 text column describing the issue. This column is free text, written by human.",
          "Yes, there are some text columns describing the issue. These columns are free text but not written by human.",
          "No",
          "Some additional option"], ""],
        [3, "Are there tagged columns describing the failed System / Subsystem / Components / Parts?", "Yes/No", ""],
        [4, "Do they already have a taxonomy to classify the events (especially categorical columns)?", "Yes/No", ""],
        [5, "If answer to 4 is Yes, are these columns fully populated with an actionable entity?", "Yes/No", ""],
        [6, "If answer to 4 is Yes, are these tags precise (wherever actionable)?", "Yes/No", ""],
        [7, "If answer to 4 is Yes, are these tags accurate (wherever actionable)?", "Yes/No", ""],
        [8, "Is Date of the event present?", "Yes/No", ""],
        [9, "Is Manufacturing date present?", "Yes/No", ""],
        [10, "Is there a supplier column?", "Yes/No", ""],
        [11, "Is there a plant column?", "Yes/No", ""],
        [12, "Is there a countermeasure column?", "Yes/No", ""],
        [13, "Is there any fix/resolution column?", "Yes/No", ""],
        [14, "Is there any identified root cause condition/components column?", "Yes/No", ""],
        [15, "Are there other interesting columns not covered by the other questions?", "Yes/No", ""],
        # Add more questions here
        [16, "Conceptual proposal", ["N/A"], ""]
    ]

    # Create the checklist DataFrame
    checklist_df = pd.DataFrame(checklist_data, columns=["#", "Questions", "Options", "Extra details"])

    # Display the checklist DataFrame as editable
    st.subheader("Checklist Table")

    # Use a dictionary to store user inputs
    user_inputs = {}

    for index, row in checklist_df.iterrows():
        st.subheader(f"Question {index + 1}: {row['Questions']}")
        # Get options from the DataFrame
        options = row["Options"]
        # Ensure options are a list
        if isinstance(options, str):
            options = options.split('\n')  # Assuming options are separated by new lines
        answer = st.selectbox("Select Answer", options=options, key=f"select_{index}")
        extra_detail = st.text_input(f"Extra details for question {index + 1}", value=row["Extra details"],
                                     key=f"extra_{index}")
        user_inputs[row["Questions"]] = {
            "answer": answer,
            "extra_details": extra_detail
        }

    # Create a form for the checklist
    with st.form("checklist_form"):
        # Submit button
        submitted = st.form_submit_button("Submit")

        if submitted:
            # Store the answers in session_state
            st.session_state.checklist_answers = user_inputs
            st.success("Checklist submitted successfully!")

    # Display the checklist answers
    if 'checklist_answers' in st.session_state:
        st.subheader("Checklist Answers")
        answers_df = pd.DataFrame.from_dict(st.session_state.checklist_answers, orient='index')
        answers_df.index.name = 'Question'
        answers_df.reset_index(inplace=True)
        st.dataframe(answers_df)


# Example usage (ensure to call the function with a DataFrame when using it in your app):
# sample_data = {"Column Name": ["Column1", "Column2"], "Column Description": ["Description1", "Description2"]}
# input_df = pd.DataFrame(sample_data)
# checklist(input_df)

def data_validated():
    st.title("Data Validated")

    # Display Home Page Data
    if 'home_data' in st.session_state and st.session_state['home_data']:
        home_data = st.session_state['home_data']
        st.subheader("Home Page Data")
        home_df = pd.DataFrame.from_dict(home_data, orient='index', columns=['Value'])
        home_df.index.name = 'Field'
        home_df.reset_index(inplace=True)
        st.dataframe(home_df)
    else:
        st.warning("No home page data available to display")

    # Display Checklist Data
    if 'checklist_answers' in st.session_state:
        st.subheader("Checklist Data")
        checklist_answers = st.session_state['checklist_answers']
        checklist_df = pd.DataFrame.from_dict(checklist_answers, orient='index')
        checklist_df.index.name = 'Question'
        checklist_df.reset_index(inplace=True)
        st.dataframe(checklist_df)
    else:
        st.warning("No checklist data available to display")

    # Display Column-Wise Analysis Data
    if 'column_wise_analysis_data' in st.session_state and st.session_state['column_wise_analysis_data'] is not None:
        column_wise_data = st.session_state['column_wise_analysis_data']
        st.subheader("Column-Wise Analysis Data")
        st.dataframe(column_wise_data)
    else:
        st.warning("No column-wise analysis data available to display")

        # Export button
        if st.button("Prepare CSV"):
            if 'home_data' in st.session_state and 'checklist_answers' in st.session_state and 'column_wise_analysis_data' in st.session_state:
                # Prepare DataFrames
                home_df = pd.DataFrame.from_dict(st.session_state['home_data'], orient='index', columns=['Value'])
                home_df.index.name = 'Field'
                home_df.reset_index(inplace=True)

                checklist_df = pd.DataFrame.from_dict(st.session_state['checklist_answers'], orient='index')
                checklist_df.index.name = 'Question'
                checklist_df.reset_index(inplace=True)

                column_wise_data = st.session_state['column_wise_analysis_data']

                # Create a CSV buffer to hold the CSV data
                buffer = io.StringIO()
                home_df.to_csv(buffer, index=False, header=True)
                buffer.write("\n\n")  # Add a separator between sections
                checklist_df.to_csv(buffer, index=False, header=True)
                buffer.write("\n\n")  # Add a separator between sections
                column_wise_data.to_csv(buffer, index=False, header=True)

                buffer.seek(0)
                st.download_button(
                    label="Download All Data as CSV",
                    data=buffer.getvalue(),
                    file_name="data_validated.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    pd.set_option('expand_frame_repr', False)  # Add this line to prevent excessive output
    main()
