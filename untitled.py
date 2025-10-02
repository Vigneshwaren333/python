import pickle
import pandas as pd
import numpy as np 
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.title("Data Analysis & File Handling GUI")

st.sidebar.title("navigation")
choice=st.sidebar.radio("Go to",["File handling","data analysis","Visualization","Numpy calculation","Ml classification"])



if not "df" in st.session_state:
    st.session_state.df = None

if choice == "File handling":
    st.header("File uploading")
    file_type = st.selectbox("Select file type",["txt","csv","xml","pkl"])
    uploaded_file = st.file_uploader("Upload a file",type=[file_type])

    mode = st.selectbox("select mode",["read","write","append"])

    if uploaded_file is not None:
        if mode == "read":
            if file_type == "csv":
                st.session_state.df = pd.read_csv(uploaded_file)
            elif file_type == "txt":
                st.session_state.df = pd.read_csv(uploaded_file,sep="\t")
            elif file_type == "xml":
                st.session_state.df = pd.read_xml(uploaded_file)
            elif file_type == "pkl":
                st.session_state.df = pd.read_pickle(uploaded_file)
            
            st.success("file read successfully")
            st.dataframe(st.session_state.df)
        elif mode == "write":
            content = st.text_area("Enter content to write")
            if st.button("Write to file"):
                try:
                    if file_type in ["csv","txt"]:
                        
                        st.session_state.df = pd.read_csv(StringIO(content))
                    elif file_type == "xml":
                       
                        st.session_state.df = pd.read_xml(StringIO(content))
                    elif file_type == "pkl":
                        st.session_state.df = pd.DataFrame(eval(content))
                   
                    st.success("File written successfully")
                    st.dataframe(st.session_state.df)
                except Exception as e:
                    st.error(f"Error writing to file: {e}")

                    if file_type in["csv","txt"]:
                        st.download_button(
                            Label = "Download file",
                            data = st.session_state.df.to_csv(index=False).encode('utf-8'),
                            file_name=uploaded_file.name+".csv",
                            mime="text/csv"
                        )
                    elif file_type == "xml":st.download_button(
                        Label= "Download file",
                        data =st.session_state.df.to_xml(index=False).encode('utf-8'),
                        file_name=uploaded_file.name+".xml",
                    )
                    elif file_type=='pkl':
                        st.download_button(
                            Label = "Download file",
                            data = pickle.dump(st.session_state.df),
                            file_name=uploaded_file.name+".pkl",
                        )

        elif mode == "append":
            content = st.text_area("Enter content to append")
            if st.button("Append to file"):
                try:
                    if file_type in ["csv","txt"]:
                        new_data = pd.read_csv(StringIO(content))
                        st.session_state.df = pd.concat([st.session_state.df,new_data])
                        st.dataframe(st.session_state.df)
                    elif file_type == "xml":
                        new_data = pd.read_xml(StringIO(content))
                        st.session_state.df = pd.concat([st.session_state.df,new_data])
                        st.dataframe(st.session_state.df)
                    elif file_type == "pkl":
                        new_data = pd.DataFrame(eval(content))
                        st.session_state.df = pd.concat([st.session_state.df,new_data])
                        st.dataframe(st.session_state.df)
                    st.success("File appended successfully")
                except Exception as e:
                    st.error(f"Error appending to file: {e}")

                #------------------ DOWNLOAD BUTTON -----------------
                if file_type in["csv","txt"]:
                        st.download_button(
                            Label = "Download file",
                            data = st.session_state.df.to_csv(index=False).encode('utf-8'),
                            file_name=uploaded_file.name+".csv",
                            mime="text/csv"
                        )
                elif file_type == "xml":st.download_button(
                        Label= "Download file",
                        data =st.session_state.df.to_xml(index=False).encode('utf-8'),
                        file_name=uploaded_file.name+".xml",
                    )
                elif file_type=='pkl':
                        st.download_button(
                            Label = "Download file",
                            data = pickle.dump(st.session_state.df),
                            file_name=uploaded_file.name+".pkl",
                        )

    if st.session_state.df is not None:
        st.subheader("Current DataFrame")
        st.dataframe(st.session_state.df)


if choice == "data analysis":
    st.header("Data analysis")
    df = st.session_state.df
    st.subheader("Dataframe")
    st.dataframe(df.info())
    st.dataframe(df.head())
    st.dataframe(df.describe())
    st.dataframe(df.tail())

    #-----Missing values-----
    st.subheader("Missing values")
    if df.isnull().sum().any():
        st.warning("Dataframe contains missing values")
        if st.button("fill the missing values"):
            df.fillna(df.mean(numeric_only=True),inplace=True)
            st.success("Missing values filled")
            st.dataframe(df)

    #-----Sorting

    sort_by = st.selectbox("Slect a column to sort",df.columns)
    st.dataframe(df.sort_values(by=sort_by))

    #-----Filtering-----
    st.subheader("Filtering")
    if len(df.select_dtypes(include = ["number"]).columns) > 0:
        col_filter = st.selectbox("Select column to filter", df.select_dtypes(include="number").columns)
        threshold = st.slider(f"show rows where {col_filter} is greater than", min_value=float(df[col_filter].min()), max_value=float(df[col_filter].max()))
        filtered_df = df[df[col_filter] > threshold]
        st.dataframe(filtered_df)
    

    #---- Grouping----

    if len(df.columns)>2:
        group_by = st.selectbox("Select column to group by", df.columns)
        st.dataframe(df.groupby(group_by).mean(numeric_only=True))


    #----Indexing/Slicing----
    st.subheader("Indexing/Slicing")
    start_idx =st.number_input("Start row index")
    end_idx = st.number_input("End row index")
    if st.button("Show sliced dataframe"):
        if 0 <= start_idx < end_idx <= len(df):
            st.dataframe(df.iloc[int(start_idx):int(end_idx)])
        else:
            st.error("Invalid indices")
        
elif choice =="Visualization":
    st.header("Visualization")

    df = st.session_state.df
    chart_type = st.selectbox("Select chart type",["line","bar","histogram","scatter","boxplot"])

    if chart_type == "line":
        cols = st.multiselect("Select columns for line chart", df.select_dtypes(include="number").columns)
        st.line_chart(df[cols])
    elif chart_type == "bar":
        cols = st.multiselect("Select columns for line chart", df.select_dtypes(include="number").columns)
        st.bar_chart(df)
    elif chart_type == "histogram":
        cols = st.multiselect("Select columns for histogram", df.select_dtypes(include="number").columns)
        fig,ax = plt.subplots()
        ax.hist(df[cols],bins=20,color ='blue',edgecolor='black')
        st.pyplot(fig)
    elif chart_type == "scatter":
        x_col = st.selectbox("Select x-axis column", df.select_dtypes(include="number").columns)
        y_col = st.selectbox("Select y-axis column", df.select_dtypes(include="number").columns)
        fig, ax = plt.subplots()
        ax.scatter(df[x_col], df[y_col], color='blue', edgecolor='black')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        st.pyplot(fig)
    elif chart_type == "boxplot":
        cols = st.multiselect("Select columns for boxplot", df.select_dtypes(include="number").columns)
        fig, ax = plt.subplots()
        ax.boxplot([df[col].dropna() for col in cols], labels=cols)
        ax.set_title("Boxplot")
        st.pyplot(fig)

elif choice == "Numpy calculation":
    st.header("Numpy calculation")
    st.subheader("basic calculatiom")


    selected_cols = st.multiselect("Select columns for calculation", st.session_state.df.select_dtypes(include="number").columns)

    if st.button("mean"):
        st.write("Mean:",st.session_state.df[selected_cols].mean())
    if st.button("median"):
        st.write("Median:", st.session_state.df[selected_cols].median())
    if st.button("std"):
        st.write("Standard Deviation:", st.session_state.df[selected_cols].std())   
    if st.button("var"):
        st.write("Variance:", st.session_state.df[selected_cols].var())
    if st.button("sum"):
        st.write("Sum:", st.session_state.df[selected_cols].sum())
    if st.button("min"):
        st.write("Minimum:", st.session_state.df[selected_cols].min())
    if st.button("max"):
        st.write("Maximum:", st.session_state.df[selected_cols].max())

# ------------------- ML CLASSIFICATION (KNN) -------------------
elif choice == "Ml classification":
    st.subheader("KNN Classification")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Clean data
        df = df.drop_duplicates()
        df = df.fillna(df.mean(numeric_only=True))
        df = df.fillna(df.mode().iloc[0])

        # Encode categorical columns
        from sklearn.preprocessing import LabelEncoder
        for col in df.select_dtypes(include="object").columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        numeric_cols = df.select_dtypes(include="number").columns
        target_col = st.selectbox("Select target variable", df.columns)
        feature_cols = [c for c in numeric_cols if c != target_col]

        st.write("Using features:", feature_cols)

        if st.button("Train KNN"):
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.metrics import accuracy_score, confusion_matrix

                X = df[feature_cols]
                y = df[target_col]

               

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test= scaler.transform(X_test)
                k_values = [1, 5, 7, 9, 10]
                accuracies = []

                for k in k_values:
                    knn = KNeighborsClassifier(n_neighbors=k)
                    knn.fit(X_train, y_train)
                    y_pred = knn.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    accuracies.append(accuracy)
                    st.write(f"k = {k}, Accuracy = {accuracy:.3f}")

                best_k = k_values[np.argmax(accuracies)]
                best_knn = KNeighborsClassifier(n_neighbors=best_k)
                best_knn.fit(X_train, y_train)
                y_pred = best_knn.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                st.write(f"✅ Best K = {best_k}, Accuracy = {accuracy:.3f}")
                st.write("Confusion Matrix:")
                st.write(cm)

            except Exception as e:
                st.error(f"Error in training KNN: {e}")
