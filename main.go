package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/labstack/echo/v4"

	"github.com/ring-c/go-web-diff/pkg/generate"
)

func main() {
	defer func() {
		if r := recover(); r != nil {
			println("Recovered panic")
			spew.Dump(r)
		}
	}()
	spew.Config.Indent = "\t"

	// Setup
	e := echo.New()
	e.HideBanner = true

	generate.Routes(e)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	// Start server
	go func() {
		if err := e.Start(":8080"); err != nil && err != http.ErrServerClosed {
			e.Logger.Fatal("shutting down the server")
		}
	}()

	// Wait for interrupt signal to gracefully shutdown the server with a timeout of 10 seconds.
	<-ctx.Done()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := e.Shutdown(ctx); err != nil {
		e.Logger.Fatal(err)
	}

	generate.ModelClose()
}
