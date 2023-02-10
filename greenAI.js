// This example requires the Drawing library. Include the libraries=drawing
// parameter when you first load the API. For example:
// <script src="https://maps.googleapis.com/maps/api/js?key=YOUR_API_KEY&libraries=drawing">
var showImgtoggle = false
var map;

function initMap() {
    map = new google.maps.Map(document.getElementById("map"), {
      center: { lat: 59.27, lng: 15.220 },
      zoom: 14.2,
    });
    var drawingManager = new google.maps.drawing.DrawingManager({
      drawingControl: true,
      drawingControlOptions: {
        position: google.maps.ControlPosition.TOP_CENTER,
        drawingModes: [
          google.maps.drawing.OverlayType.POLYGON,
        ],
      },

      polygonOptions: {
        // editable: true,
        fillOpacity: 0.7

      }
    });
  
    drawingManager.setMap(map);

    google.maps.event.addListener(drawingManager, 'polygoncomplete', function(polygon) {

        customLabel()

        var corners= []; 

        var len = polygon.getPath().getLength();

        var lats = []
        var lngs = []

        for (var i = 0; i < len; i++) {
            var corner = polygon.getPath().getAt(i).toJSON(5); //toUrlValue(5)
            
            proj4.defs([
                ['WGS84', "+title=WGS 84 (long/lat) +proj=longlat +ellps=WGS84 +datum=WGS84 +units=degrees"],
                ['SWEREF991500',"+proj=tmerc +lat_0=0 +lon_0=15 +k=1 +x_0=150000 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"]
                // ['EPSG:3009',"+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"]
               ]);

            // proj4.defs("EPSG:3009","+proj=utm +zone=33 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs");

            var epsg3009 = proj4("SWEREF991500");
            // var epsg3009 = proj4("EPSG:3009");

            lats.push(corner.lat)
            lngs.push(corner.lng)


            console.log(corner.lat, corner.lng)

            var sweref_corner = proj4('WGS84', epsg3009, [corner.lng, corner.lat]);
            
            // corners.push([sweref_corner[0] - 349986.861, sweref_corner[1] +2626.8933])
            corners.push(sweref_corner.reverse())

        }
        var gai = get_gai(corners, drawingManager);

        gai.then(value => {

            console.log(value)

            var center = polygonCenter(lngs, lats)

            console.log(center[0], center[1])


            customLabel(value, center[0], center[1])

            // var polygonOptions = drawingManager.get('polygonOptions');
            polygon.set('fillColor', perc2color(value*100, 0, 90));
            // polygon.set('strokeColor', perc2color(value*100, 10, 70));

            // drawingManager.set('polygonOptions', polygonOptions);

        })

        if (showImgtoggle == true){
            showImg
        }
        
        });

    
    }


function polygonCenter(lngs, lats) {
    // const vertices = poly.getPath();

    // put all latitudes and longitudes in arrays
    const longitudes = lngs
    const latitudes = lats;

    // sort the arrays low to high
    latitudes.sort();
    longitudes.sort();

    // get the min and max of each
    const lowX = latitudes[0];
    const highX = latitudes[latitudes.length - 1];
    const lowy = longitudes[0];
    const highy = longitudes[latitudes.length - 1];

    // center of the polygon is the starting point plus the midpoint
    const centerX = lowX + ((highX - lowX) / 2);
    const centerY = lowy + ((highy - lowy) / 2);

    return [centerX, centerY];
}

function customLabel(gai = 0.5, Lat, Lng){
    if (document.getElementById('map')){

        var markerLatLng = new google.maps.LatLng(Lat, Lng);
        // var markerLatLng = markerLatLng   
        // console.log(markerLatLng)
      
        // var mapOptions = {
        //   zoom: 16,
        //   center: markerLatLng,
        //   mapTypeId: google.maps.MapTypeId.ROADMAP
        // };
        
        // var map = new google.maps.Map(document.getElementById("map"), mapOptions);
      
        var markerIcon = {
              url: 'https://developers.google.com/maps/documentation/javascript/examples/full/images/beachflag.png',
              scaledSize: new google.maps.Size(0, 0),
              origin: new google.maps.Point(0, 0),
              anchor: new google.maps.Point(0,0),
              labelOrigin:  new google.maps.Point(0,0),
            };
        
        var markerLabel = gai;
        var marker = new google.maps.Marker({
          map: map,
          animation: google.maps.Animation.DROP,
          position: markerLatLng,
          icon: markerIcon,
          label: {
            text: markerLabel.toFixed(2).toString(),
            color: "black",
            // fontSize: "16px",
            // Size: 10,
            fontWeight: "bold",
          }
        });
       
      }
}


  

function showImg(){
    var path = '../../area_of_interest.png'
    var mask = "../../pred.png"

    document.getElementById("myMask").src = mask;
    document.getElementById("myImg").src = path;

    var toggled = false

    if(toggled == false) {
        $('#slide_in').toggleClass('show');
        $('#map').toggleClass('show');
        toggled = true
    }
};

window.initMap = initMap;

function findCenter(markers) {
    let lat = 0;
    let lng = 0;
    
    for(let i = 0; i < markers.length; ++i) {
        lat += markers[i].lat;
        lng += markers[i].lng;
    }

    lat /= markers.length;
    lng /= markers.length;

    return {lat: lat, lng: lng}
}

function perc2color(perc,min,max) {
    var base = (max - min);

    if (base == 0) { perc = 100; }
    else {
        perc = (perc - min) / base * 100; 
    }
    var r, g, b = 0;
    if (perc < 50) {
        r = 255;
        g = Math.round(5.1 * perc);
    }
    else {
        g = 255;
        r = Math.round(510 - 5.10 * perc);
    }
    var h = r * 0x10000 + g * 0x100 + b * 0x1;
    return '#' + ('000000' + h.toString(16)).slice(-6);
}
function get_gai(coords, drawingManager) {
    return fetch("http://127.0.0.1:5000/receiver", 
        {
            method: 'POST',
            headers: {
                'Content-type': 'application/json',
                'Accept': 'application/json'
            },
        // Strigify the payload into JSON:
        body:JSON.stringify(coords)}).then(res=>{
                if(res.ok){
                    return res.json()
                }else{
                    alert("something is wrong")
                }
            }).then(jsonResponse=>{

                let gai = jsonResponse;
                
                return gai

                // Log the response data in the console
                // console.log(cords)
            } 
            ).catch((err) => console.error(err));
}