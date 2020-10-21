

def make_station_dropdown_html(html_id, vec):
    html_vec = [f'<option value="{x}">{x}</option>'
                for x in vec]
    html = '\n'.join(html_vec)
    return f'''
          <select id="{html_id}">
                  <option value="">--</option>
                  {html}
          </select>   <br/>
    '''


