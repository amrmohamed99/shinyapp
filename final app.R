# Import libraries
library(shiny)
library(shinythemes)
library(data.table)
library(caret)
library(randomForest)
library(flexdashboard)
library(shinyLP)
library(psych)
library(GGally)
library(shinyWidgets)

# read dataset
TrainSet <- read.csv("training.csv", header = TRUE)
TrainSet <- TrainSet[,-1]
test_set <- read.csv("testing.csv", header = TRUE)
test_set <- test_set[,-1]
## ---------------------------------------------------------------------------- UI -----------------------------------

ui <- fluidPage(theme= shinytheme("cyborg"),
                navbarPage(title = "DPapp",  id="inTabset",
                           ##Home panel ui
                           tabPanel("Home",  h1("Hi Welcome to DPapp"),
                                    strong(h4("Diabetes Prediction shiny application using machine learning, app predicts with 89 % accuracy")),
                                    br(),
                                    actionButton('next_tab',strong(h5('Click here to start',icon("paper-plane"))),
                                                 style="color: #fff; background-color: #337ab7; border-color: #2e6da4")
                                      ),

                           ##BMI Calculator panel ui
                           tabPanel("BMI Calculator",
                                    sidebarPanel(
                                      HTML("<h3>Input parameters</h3>"),
                                      sliderInput("height",
                                                  label = "Height",
                                                  value = 175,
                                                  min = 40,
                                                  max = 250),
                                      sliderInput("weight",
                                                  label = "Weight",
                                                  value = 70,
                                                  min = 20,
                                                  max = 150),

                                      actionButton("submitbutton1",
                                                   "Submit",
                                                   class = "btn btn-primary")
                                    ),
                                    mainPanel(
                                      tags$label(h3('Calculation condition')),
                                      verbatimTextOutput('text'),
                                      tableOutput('tableresult')
                                    )
                           ),

                           ##summary statistics panel ui
                           tabPanel("Summary Statistics",
                                    sidebarLayout(
                                      sidebarPanel(
                                        fileInput("file1", "Choose CSV File", accept=c('text/csv', 'text/comma-separated-values', 'text/plain', '.csv')),
                                        selectInput(inputId = "cols1",label = "Choose Variable:", choices = "", selected = " ", multiple = TRUE),
                                        radioButtons("ssoption", "Select Option", choices = c("Summary", "Length", "Type of", "Class")),
                                        hr()

                                      ),
                                      mainPanel(tableOutput("tab1") ,
                                                fluidRow(
                                                  h3("Summary Statistics"),
                                                  div(
                                                    verbatimTextOutput("summar")
                                                  )
                                                )
                                      )
                                    )
                           ),

                           ##Visualization panel ui
                           tabPanel('Visualization',
                                    headerPanel("Interactive Data Visualization"), h4("Explore Data:"),
                                    sidebarLayout(
                                      sidebarPanel(
                                        HTML("<h4>Histogram & BoxPlot</h4>"),
                                        radioButtons("p", "Select column of diabetes dataset:",
                                                     list("Pregnancies"='a',
                                                          "Glucose"='b',
                                                          "BloodPressure"='c',
                                                          "SkinThickness"='d',
                                                          "Insulin"='e',
                                                          "BMI"='f',
                                                          "DiabetesPedigreeFunction"='g',
                                                          "Age"='h')),
                                        sliderInput("bins",
                                                    "Number of bins:",
                                                    min = 1,
                                                    max = 50,
                                                    value = 30)
                                      ),
                                      mainPanel(
                                        plotOutput("distPlot"),
                                        plotOutput("qqp")
                                      )   )),

                           ##correlation summary panel ui
                           tabPanel("Correlation Summary",
                                    sidebarLayout(
                                      sidebarPanel(
                                        selectInput("cols9", "Choose Variable:", choices = "", selected = " ", multiple = TRUE),
                                        selectInput("cols10", "Choose Variable:", choices = "", selected = " ", multiple = TRUE),
                                        radioButtons("cormethod", "Select Method:", choices = c("Covariance", "KarlPearson", "Spearman", "Kendals")),
                                        hr()
                                      ),
                                      mainPanel(
                                        h3("Covariance & Correlation"),
                                        verbatimTextOutput("cor_t")
                                      )
                                    )
                           ),

                           ##correlation plot panel ui
                           tabPanel('Correlation Plot',
                                    headerPanel("Diabetes Features Correlation"),
                                    sidebarLayout(
                                      sidebarPanel(width = 3,
                                        checkboxGroupInput("check", "Select Variables:",
                                                           choices = names(TrainSet),selected = names(TrainSet)),
                                      ),
                                      mainPanel(width = 9,
                                        plotOutput("corplot"))) ),

                           ##train rf model panel ui
                           tabPanel('Model Training',
                                    headerPanel("Random Forest Model"),
                           sidebarLayout(
                             sidebarPanel(
                               sliderInput(label = "Number of trees",min = 1,max = 1000, inputId = "num_tree_random_forest",value = 500),
                               sliderInput(label = "mtry",min = 1,max = 10, inputId = "mtry",value = 2),
                               actionButton("run_random_forest","Run Random Forest",style = 'color:white; background-color:red; padding:4px; font-size:120%')
                             ),
                             mainPanel(width = 8,
                                       fluidRow(column(5,tableOutput("conM1")),
                                                column(5, tableOutput("conM2")),
                                                column(5,tableOutput("conM3")),
                                                fluidRow(
                                                  column(6, plotOutput("con_Matrix")))
                                                , position = c("left", "right"), fluid = T)
                ))),

                           ##prediction panel ui
                           tabPanel("Prediction",
                                    headerPanel('Diabetes Predictor'),
                                    sidebarLayout(
                                      sidebarPanel(
                                        tags$label(h3('Input parameters')),
                                        numericInput("Pregnancies",
                                                     label = "Pregnancies",
                                                     value = median(TrainSet$Pregnancies)),
                                        numericInput("Glucose",
                                                     label = "Glucose",
                                                     value = median(TrainSet$Glucose)),
                                        numericInput("BloodPressure",
                                                     label = "BloodPressure",
                                                     value = median(TrainSet$BloodPressure)),
                                        numericInput("SkinThickness",
                                                     label = "SkinThickness",
                                                     value = median(TrainSet$SkinThickness)),
                                        numericInput("Insulin",
                                                     label = "Insulin",
                                                     value = median(TrainSet$Insulin)),
                                        numericInput("BMI",
                                                     label = "BMI",
                                                     value = median(TrainSet$BMI)),
                                        numericInput("DiabetesPedigreeFunction",
                                                     label = "DiabetesPedigreeFunction",
                                                     value = median(TrainSet$DiabetesPedigreeFunction)),
                                        numericInput("Age",
                                                     label = "Age",
                                                     value = median(TrainSet$Age)),
                                        actionButton("submitbutton", "Submit",
                                                     class = "btn btn-primary")
                                      ),
                                      mainPanel(
                                        tags$label(h3('Status/Output')), # Status/Output Text Box
                                        verbatimTextOutput('contents'),
                                        tableOutput('tabledata'), # Prediction results table
                                        plotOutput("plot", click = "plot_click"),
                                        verbatimTextOutput("info")
                                      )
                                    )
                           ),

                ##Contacts panel ui
                      tabPanel("Contact",
                                    sidebarLayout(
                                      sidebarPanel(
                                        h3("Information to contact")
                                      ),
                                      mainPanel(htmlOutput("text1"))
                                    )
                           )
                ))

####################################
#       Server                     #
####################################

server<- function(input, output, session) {
  ##Home panel server
  observeEvent(input$next_tab, {
    updateTabsetPanel(session, inputId = "inTabset", selected = "BMI Calculator")
  })
  #######################################################
  ##BMI calculator panel server
  datasetInput2 <- reactive({
    bmi <- input$weight/( (input$height/100) * (input$height/100) )
    bmi <- data.frame(bmi)
    names(bmi) <- "BMI"
    print(bmi)
  })
  #Status/Output Text Box
  output$text <- renderPrint({
    if (input$submitbutton1>0) {
      isolate("Calculation complete.")
    } else {
      return("Server is ready for calculation.")
    }
  })
  # Prediction results table
  output$tableresult <- renderTable({
    if (input$submitbutton1>0) {
      isolate(datasetInput2())
    }
  })

  #######################################################
  ##summary statistics panel server
  data_input <- reactive({
    infile <- input$file1
    req(infile)
    data.frame(read.csv(infile$datapath))
  })

  observeEvent(input$file1,{
    updateSelectInput(session, inputId = "cols", choices = names(data_input()))
  }
  )

  observeEvent(input$file1, {
    updateSelectInput(session, inputId = "cols1", choices = names(data_input()))
  }
  )

  summ <- reactive({
    var1 <- data_input()[,input$cols1]
    if (input$ssoption == "Summary"){
      su <- summary(var1)
      return(su)
    } else if (input$ssoption == "Length"){
      return(length(var1))
    } else if (input$ssoption == "Type of"){
      return(typeof(var1))
    } else if(input$ssoption == "Class"){
      return(class(var1))
    }
  })

  output$summar <- renderPrint({
    if (input$ssoption == "Summary"){
      summ()
    } else if (input$ssoption == "Length"){
      summ()
    } else if (input$ssoption == "Type of"){
      summ()
    } else if(input$ssoption == "Class"){
      summ()
    }
  })

  ###############################################
  ##visulaization panel server
  output$distPlot <- renderPlot({
    if(input$p=='a'){
      i<-1}
    else if(input$p=='b'){
      i<-2 }
    else if(input$p=='c'){
      i<-3 }
    else if(input$p=='d'){
      i<-4}
    else if(input$p=='e'){
      i<-5 }
    else if(input$p=='f'){
      i<-6 }
    else if(input$p=='g'){
      i<-7}
    else if(input$p=='h'){
      i<-8}
    x<- TrainSet[, i]
    bins <- seq(min(x), max(x), length.out = input$bins + 1)
    hist(x, breaks = bins, col = 'lightblue', border = "black")
  })
  output$qqp <- renderPlot({
    if(input$p=='a'){
      j<-1}
    else if(input$p=='b'){
      j<-2 }
    else if(input$p=='c'){
      j<-3 }
    else if(input$p=='d'){
      j<-4}
    else if(input$p=='e'){
      j<-5 }
    else if(input$p=='f'){
      j<-6 }
    else if(input$p=='g'){
      j<-7}
    else if(input$p=='h'){
      j<-8}
    y<- TrainSet[, j]
    bins <- seq(min(y), max(y), length.out = input$bins + 1)
    boxplot(y, col =  "bisque")
  })

  #######################################################
  ##correlation summary panel server
  data_input <- reactive({
    infile <- input$file1
    req(infile)
    data.frame(read.csv(infile$datapath))
  })

  observeEvent(input$file1,{
    updateSelectInput(session, inputId = "cols9", choices = names(data_input()))
  })

  observeEvent(input$file1, {
    updateSelectInput(session, inputId = "cols10", choices = names(data_input()))
  })

  cortest <- reactive({
    var1 <- data_input()[,input$cols9]
    var2 <- data_input()[,input$cols10]
    if (input$cormethod == "Covariance"){
      return(cov(var1, var2))
    } else if (input$cormethod == "KarlPearson"){
      return(cor.test(var1, var2, method = "pearson"))
    } else if(input$cormethod == "Spearman"){
      return(cor.test(var1, var2, method="spearman"))
    } else {
      return(cor.test(var1, var2, method="kendall"))
    }
  })

  output$cor_t <- renderPrint({

    cortest()
  })

  ##############################################################
  ##correlation plot panel server
  output$corplot <- renderPlot({
    ggpairs(TrainSet,columns= input$check,title="Correlation")
  })

  ##############################################################
  ##train rf model panel server
  output$conM1 <- renderTable({
    rf_model =  randomForest(factor(Outcome) ~ ., data=train_set, importance=TRUE, ntree=input$num_tree_random_forest, mtry = input$mtry, do.trace=100, proximity = TRUE)
    y_pred = predict(rf_model , newdata = test_set[-9])
    #check model accuracy using confusion matrix
    confmatrix = confusionMatrix(factor(test_set[,9]) , y_pred)
    table <- as.table(confmatrix)
    table
  })

  output$conM2 <- renderTable({
    rf_model =  randomForest(factor(Outcome) ~ ., data=train_set, importance=TRUE, ntree=input$num_tree_random_forest, mtry = input$mtry, do.trace=100, proximity = TRUE)
    y_pred = predict(rf_model , newdata = test_set[-9])
    #check model accuracy using confusion matrix
    confmatrix = confusionMatrix(factor(test_set[,9]) , y_pred)
    table1 <- as.matrix(confmatrix, what = "overall")
    table1_csv <- write.csv(table1, "table1.csv")
    table1 <- read.csv("table1.csv")
    table1
  })

  output$conM3 <- renderTable({
    rf_model =  randomForest(factor(Outcome) ~ ., data=train_set, importance=TRUE, ntree=input$num_tree_random_forest, mtry = input$mtry, do.trace=100, proximity = TRUE)
    y_pred = predict(rf_model , newdata = test_set[-9])
    #check model accuracy using confusion matrix
    confmatrix = confusionMatrix(factor(test_set[,9]) , y_pred)
    table2 <- as.matrix(confmatrix, what = "classes")
    table2_csv <- write.csv(table2, "table2.csv")
    table2 <- read.csv("table2.csv")
    table2
  })

  output$con_Matrix <- renderPlot({
    rf_model =  randomForest(factor(Outcome) ~ ., data=train_set, importance=TRUE, ntree=input$num_tree_random_forest, mtry = input$mtry, do.trace=100, proximity = TRUE)
    y_pred = predict(rf_model , newdata = test_set[-9])
    #check model accuracy using confusion matrix
    confmatrix = confusionMatrix(factor(test_set[,9]) , y_pred)
    confplot <- fourfoldplot(confmatrix$table, color = c("cyan", "pink"),
                             conf.level = 0, margin = 1, main = "Confusion Matrix")

    confplot
  })

  ##############################################################
  ##prediction panel server
  datasetInput <- reactive({
    df <- data.frame(
      Name = c("Pregnancies",
               "Glucose",
               "BloodPressure",
               "SkinThickness",
               "Insulin",
               "BMI",
               "DiabetesPedigreeFunction",
               "Age"),
      Value = as.character(c(input$Pregnancies,
                             input$Glucose,
                             input$BloodPressure,
                             input$SkinThickness,
                             input$Insulin,
                             input$BMI,
                             input$DiabetesPedigreeFunction,
                             input$Age)),
      stringsAsFactors = FALSE)

    Outcome <- 0
    df <- rbind(df, Outcome)
    input <- transpose(df)
    write.table(input,"input.csv", sep=",", quote = FALSE, row.names = FALSE, col.names = FALSE)

    test <- read.csv(paste("input", ".csv", sep=""), header = TRUE)

    Output <- data.frame(Prediction=predict(rf_model,test), round(predict(rf_model,test,type="prob"), 3))
    print(Output)
  })
  # Status/Output Text Box
  output$contents <- renderPrint({
    if (input$submitbutton>0) {
      isolate("Calculation complete.")
    } else {
      return("Server is ready for calculation.")
    }
  })
  # Prediction results table
  output$tabledata <- renderTable({
    if (input$submitbutton>0) {
      isolate(datasetInput())
    }
  })
  #varImpPlot
  output$plot <- renderPlot({varImpPlot(rf_model, bg = "skyblue", lcolor = "gray", color="darkblue")},res = 96)
  output$info <- renderPrint({
    req(input$plot_click)
    x <- round(input$plot_click$x, 2)
    y <- round(input$plot_click$y, 2)
    cat("[", x, ", ", y, "]", sep = "")
  })

#########################################################
##contacts panel server
output$text1 <- renderText({
  str1 <- paste(h3("Amr Mohamed Salah"))
  str2 <- paste(h4("E-mail: amrmo211999@gmail.com"))
  str3 <- paste(h4("Phone: +20 01270223518"))
  str33 = paste('*****************************************************************************************************')
  str4 <- paste(h3("Hager M. Khalil"))
  str5 <- paste(h4("E-mail: hagermohsen92@hotmail.com"))
  str6 <- paste(h4("Phone: +20 01156868266"))
  HTML(paste(str1, str2, str3,str33, str4, str5, str6, sep = '<br/>'))
})

#######################################################################################################################################

}


####################################
# Create the shiny app             #
####################################
shinyApp(ui = ui, server = server)



