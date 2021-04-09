/* jshint esversion : 6 */

var fileData;

const hasExtension = () => {
  let fileName = $('#upload').val();
  return (new RegExp('(' + ['csv'].join('|').replace(/\./g, '\\.') + ')$')).test(fileName);
};

const upload = () => {
  if(!hasExtension())alert('Incorrect file. Use a .csv file');
  else{
    $('#message').show();
    $('#next').show();
    $('.upld').hide();
    $('.box').show();
    data = event.target.files[0];
    Papa.parse(data, {
      header: true,
      dynamicTyping: true,
      complete: (results=>{
        let table_data = '<table class="table table-bordered table-striped" style="width:90%">'; 
        table_data += "<tr>";
        Object.keys(results.data[0]).forEach(k => {
          table_data+=`<th>${k}</th>`;
        });
        table_data += "</tr>";
        let table_slice = results.data.slice(0, 9);
        table_slice.forEach(e=>{
          table_data+='<tr>';
          Object.values(e).forEach(v => {
            table_data+=`<td>${v}</td>`;
          });
          table_data+='</tr>';
        });
        table_data += "</table>";
        $("#file_table #tbl").html(table_data);
        $('.md').hide(); 
        if($('#file_table').hasClass('col-md-7')){ 
          $('#file_table').removeClass('col-md-7'); 
        }
        $('#table').show();
        $('main .site__section').css('min-height', '30vh');
        fileData = results;
      })
    });
    }
};

const processData = () => {
  $('main .site__section').css('min-height', '100vh');
  $('#next').hide();
  $('.box').hide();
  $('#message').html('Please Wait while Engine Processes your data');
  $('#loader').show();
  $('#table').hide();
  let table_data;
  eel.run(fileData.data)(val=> {
    console.log(val);
    
    $('#loader').hide();
    $('.upld').show();
    $('#next').hide();

    if(!$.isEmptyObject(val)){
      
        table_data = `<table class="table table-bordered table-striped" style='margin:0 auto;'>`;
        table_data += `<tr>
                        <th>Aspects</th><th>Prediction</th>
                      </tr>`;                 
      
        val.aspects.forEach((element, index) => {
        table_data +=`<tr>
                        <td>${element}</td>
                        <td>${val.predictions[index]}</td>
                    </tr>`;
        });
        table_data += `</table>`;
        
        $('#message').html('Kindly view the result of the analysis');
        $("#file_table #tbl").html(table_data).css('border-right', '1px solid #fff');
        $('#file_table').addClass('col-md-7');
        
        table_data = `<table class="table table-bordered table-striped">`;
        
        Object.keys(val.insight).forEach(key => {
          table_data+= `<tr>
                            <td>${key.toString().toUpperCase()}</td>
                            <td>${val.insight[ key ]}</td>
                        </tr>`;
        });
        table_data+= '</table>';
        $('#insights_table #tbl').html(table_data);$('#insights_table').show();
        table_data = `<table class="table table-bordered table-striped">`;
        table_data+= `<tr>
                        <td>Accuracy :</td>
                        <td>${val.metrics.acc} %</td>
                      </tr>
                      <tr>
                        <td>Precision :</td>
                        <td>${val.metrics.prec} %</td>
                      </tr>
                      <tr>
                        <td>Recall :</td>
                        <td>${val.metrics.rec} %</td>
                      </tr>
                      <tr>
                        <td>F - Score :</td>
                        <td>${val.metrics.f_scr} %</td>
                      </tr>`;
        table_data+= '</table>';
        $('#metrics_table #tbl').html(table_data);$('#metrics_table').show();
        if(!$('.md').hasClass('col-md-4'))$('.md').addClass('col-md-4');
        $('.md').show();
        $('#table').show();
        alert('Find your insights files in this directory : ' + val.path);
        $('main .site__section').css('min-height', '30vh');
    }else {
      $('#message').html('');
      alert('DataError.Try again with a proper text file or contact developer with this code :00x45786');
    }
  });
};