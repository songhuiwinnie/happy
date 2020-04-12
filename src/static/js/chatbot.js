$(document).ready(function(){
	$("#input-field").focus();
});


var app = new Vue({
    el: '#app',
    data: {
        message: "",
        messages: [
            {
                is_system: true,
                is_link: false,
                message: "Dear valued customer, I am hui wen. How can I help you?"
            }
        ],
        response: {
            countries: [
                {
                    name: "Haha",
                    value: "haha"
                },
                {
                    name: "Gaga",
                    value: "gaga"
                }
            ]
        },
        form: {
            place_to: "",
            place_from: "",
            date_departure: "",
            date_return: ""
        }
    },
    mounted:function(){
        this.get();
    },
    methods: {
        get: function(){
            axios.get("/api/countries").then((response) => {
                this.response.countries = response.data.countries;
            });
        },
        sendMessage: function(e){
            if(e.keyCode === 13 && !e.shiftKey && this.message){
                e.preventDefault();
                this.messages.push({"message": this.message, "is_system": false, "is_link":false});
                var message = this.message;
                this.message = "";
                axios.get("/api/query", {
                    params: {
                        message: message
                    }
                }).then((response) => {
                    this.messages.push({"message": response.data.message, "is_system": true, "is_link": response.data.is_link});
                    $(".messages").animate({ scrollTop: $(document).height() }, "fast");
                    tts(response.data.message);
                    this.bindPlace(response.data);
                    this.bindDate(response.data);
                });
            }
        },
        bindPlace: function(item){
            if(item.place_from){
                this.form.place_from = item.place_from;
            }

            if(item.place_to){
                this.form.place_to = item.place_to;
            }
        },
        bindDate: function(item){
            if(item.date_departure){
                this.form.date_departure = item.date_departure;
            }

            if(item.date_return){
                this.form.date_return = item.date_return;
            }
        },
        selected: function(item, selectedItem){
            return item === selectedItem;
        }
    }
});