import gradio as gr
import pandas as pd
import altair as alt

def make_source(id):
    DATASET=pd.read_csv("data/preprocessed/outcome_n2.csv")
    DATASET=DATASET[["ID", "movid_id", "genre", "main_char", "original","action", "situation", "intention", "consequence", "S_I_A_roberta"]]
    tmp_df=DATASET[DATASET['movid_id']==id].reset_index(drop=True)
    tmp_char=list(set([x for x in list(tmp_df['main_char'])if x !='0']))
    num_seg=len(tmp_df)

    data={}

    for i in range(len(tmp_char)):
        data[tmp_char[i]]=[]
    data["Segment"]=[]

    for i in range(num_seg):
        data["Segment"].append(i+1)
        char=tmp_df['main_char'][i]
        for j in range(len(tmp_char)):
            if char == tmp_char[j]:
                data[tmp_char[j]].append(tmp_df['S_I_A_roberta'][i]+1)
            else:
                data[tmp_char[j]].append(0)

    return data, num_seg, tmp_char


def make_plot(id):
    data, num_seg, tmp_char = make_source(id)
    print(data) # segment, char_name

    main_characters=', '.join(tmp_char)
    url=f"https://www.imdb.com/title/{id}/"


    source = pd.DataFrame(data)

    color_scale = alt.Scale(
    domain=[
        0,
        1,
        2,
    ],
    range=["#cccccc", "#c30d24","#1770ab"],
    )

    y_axis = alt.Axis(title="Characters")

    plot=alt.Chart(source).transform_fold(
            tmp_char
        ).mark_rect().encode(
            x='Segment:N',
            y=alt.Y("key:N").axis(y_axis),
            color=alt.Color('value:N').title("Morality").scale(color_scale),
        ).resolve_scale(x='independent')
    
    return plot, main_characters, url

with gr.Blocks() as demo:
    movie_id=gr.Textbox(label="Movie ID")
    button = gr.Button("Submit")

    main_characters=gr.Textbox(label="Main Character List")
    url=gr.Textbox(label="Movie URL")
    plot = gr.Plot(label="Plot")

    button.click(fn=make_plot, inputs=movie_id, outputs=[plot, main_characters, url])
    demo.load(make_plot, inputs=[button], outputs=[plot])

    examples_char = gr.Examples(examples=["tt0451279", "tt0800369", "tt2262227", "tt3530002", "tt0436697", "tt2872718", "tt0780504", "tt0076759"],
                           inputs=[movie_id],
                           label="Movie Id")


if __name__ == "__main__":
    demo.launch()
