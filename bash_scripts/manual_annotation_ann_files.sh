#!/bin/bash 
for entry in "$PWD"/resume_data/*
do
  touch "${entry%.*}.ann"
done
